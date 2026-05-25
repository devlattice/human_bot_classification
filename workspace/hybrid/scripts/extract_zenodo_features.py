"""Extract chunk-level features from Zenodo human hands dataset.

Usage:
    python workspace/hybrid/scripts/extract_zenodo_features.py

Input:  workspace/dataset/source/data/zenodo_v3/poker_hands_zenodo_train.json (2.74GB, ~963k hands)
Output: workspace/hybrid/zenodo_features.parquet
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_pipeline import aggregate_chunk_from_raw_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = REPO_ROOT / "workspace" / "dataset" / "source" / "data" / "zenodo_v3" / "poker_hands_zenodo_train.json"
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "zenodo_features.parquet"
CHUNK_SIZE = 60
MIN_CHUNK_SIZE = 20


def stream_hands(path: Path):
    """Stream hands one-by-one from a JSON array file without loading all into memory."""
    import ijson
    with open(path, "rb") as f:
        parser = ijson.items(f, "item")
        for hand in parser:
            yield hand


def stream_hands_fallback(path: Path):
    """Fallback: load entire file (needs ~3GB RAM)."""
    print("Loading entire file into memory (fallback mode)...")
    with open(path) as f:
        hands = json.load(f)
    print(f"Loaded {len(hands)} hands")
    yield from hands


def main():
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE} hands")
    print()

    # Try streaming first, fallback to full load
    try:
        import ijson  # noqa: F401
        hand_iter = stream_hands(INPUT_PATH)
        streaming = True
        print("Using ijson streaming parser")
    except ImportError:
        hand_iter = stream_hands_fallback(INPUT_PATH)
        streaming = False

    rows = []
    buffer = []
    total_hands = 0
    t0 = time.time()

    for hand in hand_iter:
        buffer.append(hand)
        total_hands += 1

        if len(buffer) >= CHUNK_SIZE:
            features = aggregate_chunk_from_raw_hands(buffer)
            features["label"] = 0  # human
            features["source"] = "zenodo"
            features["date"] = "zenodo"
            rows.append(features)
            buffer = []

            if len(rows) % 500 == 0:
                elapsed = time.time() - t0
                rate = total_hands / elapsed
                print(f"  {len(rows)} chunks | {total_hands} hands | {rate:.0f} hands/s | {elapsed:.1f}s")

    # Handle remaining buffer
    if len(buffer) >= MIN_CHUNK_SIZE:
        features = aggregate_chunk_from_raw_hands(buffer)
        features["label"] = 0
        features["source"] = "zenodo"
        features["date"] = "zenodo"
        rows.append(features)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Total hands processed: {total_hands}")
    print(f"Chunks created: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"other_ratio_mean: mean={df['other_ratio_mean'].mean():.6f} (should be 0)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
