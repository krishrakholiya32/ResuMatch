"""Evaluate a trained resume-fit checkpoint against the held-out test split.

Usage:
    python evaluate.py --config ../configs/train_config.yaml \
        --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from train import LABELS, ResumeFitDataset, collate, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = select_device()

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.checkpoint)
    model = DistilBertForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()

    df = pd.read_csv(Path(cfg["data_dir"]) / f"{args.split}.csv")
    ds = ResumeFitDataset(df, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"])
    collate_fn = lambda batch: collate(batch, tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device)).logits
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"split={args.split} n={len(all_labels)} macro_f1={macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=LABELS, columns=LABELS))


if __name__ == "__main__":
    main()
