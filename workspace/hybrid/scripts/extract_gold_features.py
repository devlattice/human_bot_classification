"""Extract chunk-level features from Gold dataset (validator evaluation data).

Usage:
    python workspace/hybrid/scripts/extract_gold_features.py

Input:  workspace/dataset/source/gold_dataset/2026-*.json
Output:
  workspace/hybrid/dataset/train/gold_features.parquet       (Apr30–May7)
  workspace/hybrid/dataset/test/may8_gold_test_features.parquet  (May-8 hold-out)
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_pipeline import aggregate_chunk_from_raw_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO_ROOT / "workspace" / "dataset" / "source" / "gold_dataset"
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
MAY8_TEST_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test" / "may8_gold_test_features.parquet"
MAY8_DATE = "2026-05-08"


def main():
    files = sorted(GOLD_DIR.glob("2026-*.json"))
    print(f"Gold files found: {len(files)}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    rows = []
    for f in files:
        date = f.stem
        with open(f) as fh:
            data = json.load(fh)

        day_count = 0
        for entry in data["data"]["chunks"]:
            inner_chunks = entry.get("chunks", [])
            labels = entry.get("groundTruth", [])

            for chunk_hands, label in zip(inner_chunks, labels):
                features = aggregate_chunk_from_raw_hands(chunk_hands)
                features["label"] = label  # 0=human, 1=bot
                features["source"] = "gold"
                features["date"] = date
                rows.append(features)
                day_count += 1

        print(f"  {date}: {day_count} chunks")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAY8_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    may8 = df[df["date"].astype(str) == MAY8_DATE].copy()
    train = df[df["date"].astype(str) != MAY8_DATE].copy()
    train.to_parquet(OUTPUT_PATH, index=False)
    if len(may8):
        may8.to_parquet(MAY8_TEST_PATH, index=False)

    print()
    print(f"Total chunks: {len(df)}")
    print(f"  Human (label=0): {(df['label']==0).sum()}")
    print(f"  Bot   (label=1): {(df['label']==1).sum()}")
    print(f"  Columns: {len(df.columns)}")
    print(f"Saved train (Apr30–May7): {OUTPUT_PATH}  rows={len(train)}")
    if len(may8):
        print(f"Saved test  (May-8):      {MAY8_TEST_PATH}  rows={len(may8)}")
    else:
        print(f"[warn] no rows for {MAY8_DATE}; test file not written")


if __name__ == "__main__":
    main()
