"""Extract chunk-level features from Gold dataset (validator evaluation data).

Usage:
    python workspace/hybrid/scripts/extract_gold_features.py

Input:  workspace/dataset/source/gold_dataset/2026-*.json
Output: workspace/hybrid/gold_features.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from poker44.validator.chunk_features import aggregate_chunk_from_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO_ROOT / "workspace" / "dataset" / "source" / "gold_dataset"
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"


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
                features = aggregate_chunk_from_hands(chunk_hands, skip_sanitize=False)
                features["label"] = label  # 0=human, 1=bot
                features["source"] = "gold"
                features["date"] = date
                rows.append(features)
                day_count += 1

        print(f"  {date}: {day_count} chunks")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Total chunks: {len(df)}")
    print(f"  Human (label=0): {(df['label']==0).sum()}")
    print(f"  Bot   (label=1): {(df['label']==1).sum()}")
    print(f"  Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
