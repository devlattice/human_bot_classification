"""Extract chunk-level features from public human hands dataset.

Usage:
    python workspace/hybrid/scripts/extract_public_features.py

Input:  workspace/dataset/source/data/poker_hands_train.json (93MB, ~25k hands)
Output: workspace/hybrid/public_features.parquet
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_pipeline import aggregate_chunk_from_raw_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = REPO_ROOT / "workspace" / "dataset" / "source" / "data" / "poker_hands_train.json"
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "public_features.parquet"
CHUNK_SIZE = 60
MIN_CHUNK_SIZE = 20


def main():
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE} hands")
    print()

    print("Loading public hands...")
    with open(INPUT_PATH) as f:
        hands = json.load(f)
    print(f"Loaded {len(hands)} hands")

    chunks = [hands[i:i + CHUNK_SIZE] for i in range(0, len(hands), CHUNK_SIZE)]
    if len(chunks[-1]) < MIN_CHUNK_SIZE:
        chunks = chunks[:-1]
    print(f"Chunks created: {len(chunks)}")

    t0 = time.time()
    rows = []
    for i, chunk_hands in enumerate(chunks):
        features = aggregate_chunk_from_raw_hands(chunk_hands)
        features["label"] = 0  # all human
        features["source"] = "public"
        features["date"] = "public"
        rows.append(features)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(chunks)} chunks...")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Total chunks: {len(df)} (all human)")
    print(f"  Columns: {len(df.columns)}")
    print(f"  other_ratio_mean: {df['other_ratio_mean'].mean():.6f} (should be 0)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
