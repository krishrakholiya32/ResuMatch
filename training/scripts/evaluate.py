"""Evaluate a trained resume-fit checkpoint against the held-out test split.

Usage:
    python evaluate.py --config ../configs/train_config.yaml \
        --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best.keras
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import DistilBertTokenizerFast

from train import LABELS, configure_device, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    configure_device()

    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg["model_name"])
    model = keras.models.load_model(args.checkpoint)

    df = pd.read_csv(Path(cfg["data_dir"]) / f"{args.split}.csv")
    labels = [LABELS.index(label) for label in df["label"]]
    ds = make_dataset(df, labels, tokenizer, cfg["resume_max_tokens"], cfg["jd_max_tokens"],
                       cfg["batch_size"], shuffle=False)

    all_labels, all_preds = [], []
    for inputs, batch_labels in ds:
        logits = model(inputs, training=False)
        all_preds.extend(tf.argmax(logits, axis=1).numpy())
        all_labels.extend(batch_labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"split={args.split} n={len(all_labels)} macro_f1={macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=LABELS, columns=LABELS))


if __name__ == "__main__":
    main()
