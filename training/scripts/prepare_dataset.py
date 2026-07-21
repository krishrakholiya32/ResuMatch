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

Collapsed to 2 classes: the original 3-way label is merged to No Fit / Fit (Potential Fit and
Good Fit combined). This isn't just a score-boosting move -- "Potential Fit" and "Good Fit"
lead to the same real user action (apply), so the 3-way split wasn't adding decision-relevant
signal, just a fuzzier boundary to get wrong. Merging also happens to roughly balance the
classes (original label mix is ~50/25/25), which trains more stably too.

Excludes 35 confirmed-mislabeled rows (EXCLUDED_PAIR_HASHES below) from train+val only --
test.csv is left exactly as published, since it wasn't part of this audit and should stay
comparable. Found via: a blind 30-row manual audit (read resume+JD, judge independently,
*then* compare to the published label) showed only ~67% human/label agreement -- a hard
ceiling on model accuracy, since you can't train past your labels' own self-consistency.
That motivated a systematic pass: TF-IDF cosine similarity between resume_text and
job_description_text as a label-independent "topical overlap" signal, used to surface the
~2% most suspicious "Fit"-labeled rows (near-zero overlap despite being labeled a match) plus
a smaller "No Fit"-labeled high-overlap check. All 90 candidates were read and judged by hand
(same bar as the blind audit -- only confident errors flagged, not borderline calls); 35 were
confirmed genuine mismatches (e.g. a Cook/Dishwasher resume labeled Fit for a 10+-year senior
Salesforce Architect role) rather than defensible edge cases. This was a targeted worst-case
search, not a random sample, so 35/90 does NOT estimate the full dataset's error rate --
treat this as a conservative, hand-verified floor (0.56% of train+val), not a complete clean.

Usage (Kaggle notebook, internet enabled):
    python prepare_dataset.py --output_dir /kaggle/working/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit

REPO_ID = "cnamuangtoun/resume-job-description-fit"

# Raw labels as published by the dataset -- validated against before merging, so a change in
# the source data's label set doesn't silently pass through the merge as a new class.
RAW_LABELS = ["No Fit", "Potential Fit", "Good Fit"]
LABEL_MERGE_MAP = {"No Fit": "No Fit", "Potential Fit": "Fit", "Good Fit": "Fit"}
LABELS = ["No Fit", "Fit"]

# sha256(resume_text + "||" + job_description_text) for each confirmed-mislabeled row -- see
# the module docstring for how these were found and verified. Content-hash keyed (not
# row-index keyed) so this stays correct even if the upstream dataset's row order ever shifts.
EXCLUDED_PAIR_HASHES = {
    "01284097af79f3ef819c55a356e130e17decd2c3eea1ac50ec349cea11a69864",
    "17c16f179ad8bde7b519b12ae867e0a19b2bec095bfa79b809b22b8ddb6db5f0",
    "1b5441346aaf895ac19f0a8c2b93e6c66e1543fdc257650766dd04c9018b6dd0",
    "2057430cab182fa5331fcac381688b350e90449b44556b9db20c306d92c91c16",
    "27dd42529e64915aa9a9d896df94e4d9b22b5bb72a2badb7a03d99a5e8834e7c",
    "289899607461e76e8d638760a9b932f50d68bec8615ee8063d504f1d01a59066",
    "34f8ee1169bb240c9245a1d7d3aafaa9f1b27e821596d97b34355a655010cfa1",
    "36308e89aaa81883439e698ad2c3d7aada6d8eb0b8382a9178e5108a0258ad87",
    "3bc59079e9e50dcc7cd378c5f91c6cec0e58b6fe0c13fffa957cc8eabe4bc992",
    "3c7e2deeb8b6b928d1ed5f9ead3e677b7899ea91e40416e72e52f597c20d4705",
    "3ef2e30eeed50d8b0347afef874f2d1aa7b45518bce1dce5cd25a03e07e6f6c7",
    "4d2c2476c0751ae867b382ae813207c18004cf058121add09c929232c8a67d2d",
    "4e040d745ba0b4dfb09a65d8e5050f2a364f22febb7cf998a4f7a39865adffe8",
    "52b8e43d9e2d5e2e789dfa2f1faf842897c449d24d008a09c652e696876b6b5f",
    "555fbc9c1fb92eebaa0b4e3c9fbd8bd1505b7a5e3ac1d3474d3cc2e6b540e799",
    "5b6d76ff7cde8b1dee07ed818d69c3c647cd489ab6481d79101e515a7ebc2761",
    "6fa9fefaa802cf320ffcdb9c66d2bf85d5d0f8d84a66882b9fca1db9c0447d24",
    "8bbcec64f91780fd288501c485c265b9e81f205308b3cd946eab4b465cba2f41",
    "a005e9f801e453d0fcc25b230891c0a346a17498060f8d1fa92bb28b354500b4",
    "a0102dbe31984419ee4a67d37b1030b8dcfcde005398412729033bbb0849c79a",
    "ab940b68a6850954749a2ac20e18bd7981e7b8eeefc4afc460406025c6e102f3",
    "aecf5d0de4655aa17bf29a54974a6b2cbfdf9f9cd1a370bf78c26822b6d19079",
    "aed9ff400b2ea9144de3af7561587dc479c1ed14be2a147dcfed42aaa943bdf2",
    "b1011febe2c7db2ed9ef1c9815bc5757f391908bb1fa55b1c263f8f479cb952c",
    "b60079c5d4814797d498704d576c14e7b0ccd7d17073cb5caeff86ab00fad90c",
    "c61d7265a275394dc04f83b94b72b755eec3c6812d1d2d9dc87821407a78d9ac",
    "cc145b8655af881ac8a89a0f268af8c2e7426b0e6b3762eeb792d342f35bc58b",
    "d22a9e491c76761be4c3e1fa58c614020e87efba3c932a6205c5801bdfe9d8fc",
    "e429dabf2e060c119aeb61f773dd6ccfca8cf716b529c828f0113b6fe0a631d7",
    "e9b39e12f817583dd98a646163f602032fda5864359f4dc83c988d5f4e2f7d16",
    "f0041d1ea9bd9a35f21422e060d54f77654e702eb59ca2b985276cf0c0cbb086",
    "f4fba64395405633e76c7f4ffb8840b390d70473d7d6336b4d74ee9b703e2441",
    "f5c056bec49b10267408944e6e584122754fc6ddbc2393ce9b98785db53876b7",
    "f7aa473db31b173a674285c5276a66f6f3ed8b259c3ec94ad2c0ecf511e0307e",
    "ffde205ed1c7010784609360742b402fc683e194e094a87b9b52a9797ee86997",
}


def pair_hash(resume_text: str, job_description_text: str) -> str:
    return hashlib.sha256((resume_text + "||" + job_description_text).encode("utf-8")).hexdigest()


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

    assert set(full_train["label"].unique()) <= set(RAW_LABELS), "unexpected raw label values"
    assert set(test["label"].unique()) <= set(RAW_LABELS), "unexpected raw label values"
    full_train["label"] = full_train["label"].map(LABEL_MERGE_MAP)
    test["label"] = test["label"].map(LABEL_MERGE_MAP)

    # Drop confirmed-mislabeled rows from train+val only -- see module docstring. test.csv is
    # untouched (kept exactly as published) since it wasn't part of this audit.
    pair_hashes = full_train.apply(lambda r: pair_hash(r["resume_text"], r["job_description_text"]), axis=1)
    excluded_mask = pair_hashes.isin(EXCLUDED_PAIR_HASHES)
    n_excluded = int(excluded_mask.sum())
    assert n_excluded == len(EXCLUDED_PAIR_HASHES), (
        f"expected to find all {len(EXCLUDED_PAIR_HASHES)} excluded rows in train.csv, "
        f"found {n_excluded} -- upstream dataset may have changed"
    )
    full_train = full_train[~excluded_mask].reset_index(drop=True)
    print(f"excluded {n_excluded} confirmed-mislabeled rows from train+val")

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
