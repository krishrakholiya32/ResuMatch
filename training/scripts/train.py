"""Fine-tune DistilBERT as a 3-class resume<->job-description fit classifier.

Single-phase full fine-tune (unlike the image projects' freeze/unfreeze recipe --
transformer fine-tuning conventionally trains all layers at a low LR from the start).
Class-weighted loss handles the label imbalance (~50% No Fit / 25% / 25%).

Usage:
    python train.py --config ../configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup

LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


MIN_SUPPORTED_CUDA_CAPABILITY = (7, 0)  # older GPUs (e.g. Kaggle's P100, sm_60) aren't supported
# by recent PyTorch prebuilt wheels' compiled kernels; Kaggle's GPU pool mixes P100 and T4 and
# doesn't let you pin which one you get, so detect and fall back instead of crashing.


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    capability = torch.cuda.get_device_capability(0)
    if capability < MIN_SUPPORTED_CUDA_CAPABILITY:
        name = torch.cuda.get_device_name(0)
        print(f"GPU {name} has CUDA capability {capability}, below the minimum "
              f"{MIN_SUPPORTED_CUDA_CAPABILITY} this PyTorch build supports -- falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda")


class ResumeFitDataset(Dataset):
    """Tokenizes resume and JD separately with fixed per-side token budgets, then
    combines as [CLS] resume [SEP] jd [SEP] -- avoids the default pair-truncation
    strategy silently favoring whichever text happens to be shorter.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, resume_max_tokens: int, jd_max_tokens: int):
        self.resumes = df["resume_text"].tolist()
        self.jds = df["job_description_text"].tolist()
        self.labels = [LABELS.index(label) for label in df["label"]]
        self.tokenizer = tokenizer
        self.resume_max_tokens = resume_max_tokens
        self.jd_max_tokens = jd_max_tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        resume_ids = self.tokenizer.encode(
            self.resumes[idx], add_special_tokens=False, truncation=True, max_length=self.resume_max_tokens
        )
        jd_ids = self.tokenizer.encode(
            self.jds[idx], add_special_tokens=False, truncation=True, max_length=self.jd_max_tokens
        )
        input_ids = [self.tokenizer.cls_token_id] + resume_ids + [self.tokenizer.sep_token_id] + \
            jd_ids + [self.tokenizer.sep_token_id]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate(batch, pad_token_id: int):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.stack([item["label"] for item in batch])
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool, log_every: int = 25):
    model.train(mode=train)
    total_loss, n = 0.0, 0
    all_labels, all_preds = [], []
    phase = "train" if train else "val"
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())

        if step % log_every == 0 or step == len(loader):
            elapsed = time.time() - start
            print(f"  [{phase}] batch {step}/{len(loader)} "
                  f"({elapsed:.1f}s elapsed, {elapsed / step:.2f}s/batch, running_loss={total_loss / n:.4f})",
                  flush=True)

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = select_device()
    print(f"device: {device}")

    data_dir = Path(cfg["data_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])
    train_ds = ResumeFitDataset(train_df, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"])
    val_ds = ResumeFitDataset(val_df, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"])

    collate_fn = lambda batch: collate(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], collate_fn=collate_fn)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_ds.labels)
    print(f"class weights: {dict(zip(LABELS, class_weights))}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    model = DistilBertForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg["warmup_ratio"]), num_training_steps=total_steps
    )

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "labels.json", "w") as f:
        json.dump(LABELS, f, indent=2)

    # Auto-resume: a Kaggle CPU run can take hours and risks hitting the platform's session
    # time limit, so every epoch's full training state (not just the best model weights) is
    # checkpointed -- rerunning the same command picks up where it left off instead of
    # restarting from scratch.
    state_path = ckpt_dir / "last_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_macro_f1 = state["best_macro_f1"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        print(f"resuming from {state_path}: starting at epoch {start_epoch} "
              f"(best_val_macro_f1={best_macro_f1:.4f} so far)")
    else:
        start_epoch = 1
        best_macro_f1 = -1.0
        epochs_without_improvement = 0
        history = []

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, scheduler, device, train=False)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_macro_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                         "val_loss": val_loss, "val_macro_f1": val_f1})

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            epochs_without_improvement = 0
            model.save_pretrained(ckpt_dir / "best")
            tokenizer.save_pretrained(ckpt_dir / "best")
        else:
            epochs_without_improvement += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_f1": best_macro_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }, state_path)

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch} (best val_macro_f1={best_macro_f1:.4f})")
            break

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_macro_f1": best_macro_f1}, f, indent=2)

    print(f"done. best val_macro_f1={best_macro_f1:.4f}. best checkpoint: {ckpt_dir / 'best'}")


if __name__ == "__main__":
    main()
