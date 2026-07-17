"""Download the resume-JD fit dataset from Hugging Face and carve out a val split.

Source: cnamuangtoun/resume-job-description-fit (6,241 train / 1,759 test rows,
columns: resume_text, job_description_text, label in {No Fit, Potential Fit, Good Fit}).
The published test.csv is kept untouched as the final holdout. val.csv is a GROUP split
by job_description_text (not a plain row-level stratified split) -- train.csv only has 279
unique JDs reused across 5,304 rows, so a naive random split put ~all of val's JDs into
train too (just paired with different resumes), inflating val accuracy on a task the model
had effectively already seen. Grouping by JD keeps val's job descriptions fully disjoint
from train's, matching how the official test.csv is already structured, so val now measures
genuine generalization to unseen JDs instead of "new resume, already-seen JD."

Usage (Kaggle notebook, internet enabled):
    python prepare_dataset.py --output_dir /kaggle/working/data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit

REPO_ID = "cnamuangtoun/resume-job-description-fit"

# Fixed order (not alphabetical) so the label is ordinal: index reflects fit strength.
LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def download_csv(filename: str, retries: int = 5, backoff_seconds: float = 5.0) -> pd.DataFrame:
    """Uses huggingface_hub's resolution API rather than a raw GET against resolve/main/ --
    a raw GET against the CSV URL was seen to hit sustained 504s from Kaggle's network across
    every retry, while hf_hub_download hits HF's actual CDN resolution path and handles its
    own connection retries too."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
            return pd.read_csv(path)
        except Exception as e:
            last_error = e
            if attempt == retries:
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            print(f"download failed ({e}), retrying in {wait:.0f}s ({attempt}/{retries})")
            time.sleep(wait)
    raise last_error  # unreachable, satisfies type checkers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_train = download_csv("train.csv")
    test = download_csv("test.csv")
    print(f"downloaded: train={len(full_train)} test={len(test)}")

    assert set(full_train["label"].unique()) <= set(LABELS), "unexpected label values"

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(full_train, groups=full_train["job_description_text"]))
    train, val = full_train.iloc[train_idx], full_train.iloc[val_idx]

    train_jds = set(train["job_description_text"])
    val_jds = set(val["job_description_text"])
    overlap = train_jds & val_jds
    assert not overlap, f"group split failed: {len(overlap)} job descriptions leaked into both train and val"

    print(f"split: train={len(train)} val={len(val)} test={len(test)} "
          f"(train JDs={len(train_jds)}, val JDs={len(val_jds)}, overlap={len(overlap)})")

    train.to_csv(args.output_dir / "train.csv", index=False)
    val.to_csv(args.output_dir / "val.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    with open(args.output_dir / "labels.json", "w") as f:
        json.dump(LABELS, f, indent=2)
    print(f"labels.json written: {LABELS}")


if __name__ == "__main__":
    main()
