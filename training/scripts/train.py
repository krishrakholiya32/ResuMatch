"""Fine-tune DistilBERT as a 3-class resume<->job-description fit classifier.

Single-phase full fine-tune (unlike the image projects' freeze/unfreeze recipe --
transformer fine-tuning conventionally trains all layers at a low LR from the start).
Class-weighted loss handles the label imbalance (~50% No Fit / 25% / 25%).

TensorFlow/Keras via KerasHub (previously PyTorch via HuggingFace Transformers).
HuggingFace deprecated and is removing TensorFlow support from `transformers` (the TF model
classes are gone as of v5, and even the last v4.x release prints a deprecation warning and
has a broken PyTorch->TF weight-conversion path for this model) -- KerasHub is the Keras
team's own actively-maintained model library and isn't affected by that. `transformers` is
still used here, but only for its framework-agnostic fast tokenizer (no torch/TF pulled in
by that import). See export_model.py for a tf2onnx compatibility issue this pipeline also
had to work around.

Usage:
    python train.py --config ../configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizerFast

LABELS = ["No Fit", "Potential Fit", "Good Fit"]
KERAS_HUB_PRESET = "distil_bert_base_en_uncased"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_device() -> str:
    """Enable memory growth on any visible GPU. Unlike the PyTorch version this replaced,
    no compute-capability fallback is needed: TensorFlow's prebuilt pip wheels are compiled
    for a broad compute-capability range (including Kaggle's older P100s, sm_60), whereas
    recent PyTorch wheels dropped compiled-kernel support for those and needed an explicit
    CPU fallback to avoid crashing."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = "GPU" if gpus else "CPU"
    print(f"device: {device} ({len(gpus)} visible)")
    return device


def encode_pair(tokenizer, resume_text: str, jd_text: str, resume_max_tokens: int, jd_max_tokens: int) -> list[int]:
    """Tokenizes resume and JD separately with fixed per-side token budgets, then combines
    as [CLS] resume [SEP] jd [SEP] -- avoids the default pair-truncation strategy silently
    favoring whichever text happens to be shorter.
    """
    resume_ids = tokenizer.encode(
        resume_text, add_special_tokens=False, truncation=True, max_length=resume_max_tokens
    )
    jd_ids = tokenizer.encode(jd_text, add_special_tokens=False, truncation=True, max_length=jd_max_tokens)
    return [tokenizer.cls_token_id] + resume_ids + [tokenizer.sep_token_id] + jd_ids + [tokenizer.sep_token_id]


def make_dataset(df: pd.DataFrame, labels: list[int], tokenizer, resume_max_tokens: int, jd_max_tokens: int,
                  batch_size: int, shuffle: bool) -> tf.data.Dataset:
    """Dynamic per-batch padding (via padded_batch), matching the original collate_fn's
    behavior instead of padding every example to a fixed global max length. Yields
    {"token_ids", "padding_mask"} dicts -- KerasHub's DistilBertBackbone input names --
    instead of HF's {"input_ids", "attention_mask"}.
    """
    resumes = df["resume_text"].tolist()
    jds = df["job_description_text"].tolist()

    def generator():
        for resume, jd, label in zip(resumes, jds, labels):
            token_ids = encode_pair(tokenizer, resume, jd, resume_max_tokens, jd_max_tokens)
            yield {"token_ids": token_ids, "padding_mask": [1] * len(token_ids)}, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                "token_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "padding_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=0, reshuffle_each_iteration=True)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=({"token_ids": [None], "padding_mask": [None]}, []),
        padding_values=({"token_ids": tf.cast(tokenizer.pad_token_id, tf.int32), "padding_mask": 0}, 0),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


class LinearWarmupDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to `peak_lr` over `warmup_steps`, then linear decay to 0 by
    `total_steps` -- matches the shape of HF's get_linear_schedule_with_warmup (used by the
    PyTorch version this replaced) without depending on `transformers`' TF-specific
    optimization helpers.
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(max(self.warmup_steps, 1), tf.float32)
        total_steps = tf.cast(max(self.total_steps, 1), tf.float32)
        warmup_lr = self.peak_lr * (step / warmup_steps)
        decay_progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        decay_lr = self.peak_lr * tf.maximum(0.0, 1.0 - decay_progress)
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


def run_epoch(model, dataset, class_weights: tf.Tensor, optimizer, train: bool, log_every: int = 25):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    total_loss, n = 0.0, 0
    all_labels, all_preds = [], []
    phase = "train" if train else "val"
    start = time.time()
    step = 0

    for step, (inputs, labels) in enumerate(dataset, start=1):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=train)
            per_example_loss = loss_fn(labels, logits)
            sample_weights = tf.gather(class_weights, labels)
            loss = tf.reduce_mean(per_example_loss * sample_weights)
        if train:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        batch_size = int(labels.shape[0])
        total_loss += float(loss) * batch_size
        n += batch_size
        all_labels.extend(labels.numpy())
        all_preds.extend(tf.argmax(logits, axis=1).numpy())

        if step % log_every == 0:
            elapsed = time.time() - start
            print(f"  [{phase}] batch {step} "
                  f"({elapsed:.1f}s elapsed, {elapsed / step:.2f}s/batch, running_loss={total_loss / n:.4f})",
                  flush=True)

    elapsed = time.time() - start
    print(f"  [{phase}] {step} batches in {elapsed:.1f}s")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    configure_device()

    data_dir = Path(cfg["data_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    train_labels = [LABELS.index(label) for label in train_df["label"]]
    val_labels = [LABELS.index(label) for label in val_df["label"]]

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])
    train_ds = make_dataset(train_df, train_labels, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"],
                             cfg["batch_size"], shuffle=True)
    val_ds = make_dataset(val_df, val_labels, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"],
                           cfg["batch_size"], shuffle=False)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_labels)
    print(f"class weights: {dict(zip(LABELS, class_weights))}")
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    model = keras_hub.models.DistilBertClassifier.from_preset(
        KERAS_HUB_PRESET, num_classes=len(LABELS), preprocessor=None
    )

    steps_per_epoch = -(-len(train_labels) // cfg["batch_size"])  # ceil division
    total_steps = steps_per_epoch * cfg["epochs"]
    lr_schedule = LinearWarmupDecay(
        peak_lr=cfg["lr"], warmup_steps=int(total_steps * cfg["warmup_ratio"]), total_steps=total_steps
    )
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg["weight_decay"])

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "labels.json", "w") as f:
        json.dump(LABELS, f, indent=2)

    # Auto-resume: a Kaggle run can take hours and risks hitting the platform's session time
    # limit, so every epoch's full training state is checkpointed -- rerunning the same
    # command picks up where it left off instead of restarting from scratch. Unlike the
    # PyTorch version's single last_state.pt bundle, a TF checkpoint only captures
    # model+optimizer variables (not arbitrary Python state), so epoch/best-F1/history are
    # tracked in a companion JSON file saved in lockstep with each checkpoint.
    tf_ckpt_dir = ckpt_dir / "last_state"
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(tf_ckpt_dir), max_to_keep=1)
    meta_path = ckpt_dir / "last_state_meta.json"

    if manager.latest_checkpoint and meta_path.exists():
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        with open(meta_path) as f:
            meta = json.load(f)
        start_epoch = meta["epoch"] + 1
        best_macro_f1 = meta["best_macro_f1"]
        epochs_without_improvement = meta["epochs_without_improvement"]
        history = meta["history"]
        print(f"resuming from {manager.latest_checkpoint}: starting at epoch {start_epoch} "
              f"(best_val_macro_f1={best_macro_f1:.4f} so far)")
    else:
        start_epoch = 1
        best_macro_f1 = -1.0
        epochs_without_improvement = 0
        history = []

    best_path = ckpt_dir / "best.keras"
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss, train_f1 = run_epoch(model, train_ds, class_weights_tensor, optimizer, train=True)
        val_loss, val_f1 = run_epoch(model, val_ds, class_weights_tensor, optimizer, train=False)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_macro_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                         "val_loss": val_loss, "val_macro_f1": val_f1})

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            epochs_without_improvement = 0
            model.save(best_path)
        else:
            epochs_without_improvement += 1

        manager.save()
        with open(meta_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "best_macro_f1": best_macro_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            }, f, indent=2)

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"early stopping at epoch {epoch} (best val_macro_f1={best_macro_f1:.4f})")
            break

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "best_val_macro_f1": best_macro_f1}, f, indent=2)

    print(f"done. best val_macro_f1={best_macro_f1:.4f}. best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
