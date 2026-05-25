"""Extract chunk-level features for all TEST sets (unseen evaluation data).

Usage:
    python workspace/hybrid/scripts/extract_test_features.py

Inputs:
    zenodo_test:  workspace/dataset/source/data/zenodo_v3/poker_hands_zenodo_test.json  (240K human hands)
    public_test:  workspace/dataset/source/data/poker_hands_test.json                    (6.4K human hands)
    acpc_bot_test: workspace/dataset/source/data/bot/poker_hands_bot_test.json           (20K bot hands)

Outputs (all in workspace/hybrid/dataset/test/):
    zenodo_test_features.parquet
    public_test_features.parquet
    acpc_bot_test_features.parquet
    may8_gold_test_features.parquet  (from split_gold_may8_to_test / extract_gold_features)
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
OUTPUT_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test"
CHUNK_SIZE = 60
MIN_CHUNK_SIZE = 20

SOURCES = [
    {
        "name": "zenodo_test",
        "path": REPO_ROOT / "workspace" / "dataset" / "source" / "data" / "zenodo_v3" / "poker_hands_zenodo_test.json",
        "label": 0,
        "output": "zenodo_test_features.parquet",
    },
    {
        "name": "public_test",
        "path": REPO_ROOT / "workspace" / "dataset" / "source" / "data" / "poker_hands_test.json",
        "label": 0,
        "output": "public_test_features.parquet",
    },
    {
        "name": "acpc_bot_test",
        "path": REPO_ROOT / "workspace" / "dataset" / "source" / "data" / "bot" / "poker_hands_bot_test.json",
        "label": 1,
        "output": "acpc_bot_test_features.parquet",
    },
]


def extract_features(hands: list, label: int, source_name: str) -> pd.DataFrame:
    chunks = [hands[i:i + CHUNK_SIZE] for i in range(0, len(hands), CHUNK_SIZE)]
    if chunks and len(chunks[-1]) < MIN_CHUNK_SIZE:
        chunks = chunks[:-1]

    rows = []
    for i, chunk_hands in enumerate(chunks):
        features = aggregate_chunk_from_raw_hands(chunk_hands)
        features["label"] = label
        features["source"] = source_name
        rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(chunks)} chunks...")

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print()

    for src in SOURCES:
        name = src["name"]
        path = src["path"]
        label_name = "human" if src["label"] == 0 else "bot"

        print(f"{'='*60}")
        print(f"Extracting: {name} ({label_name})")
        print(f"  Input: {path}")

        if not path.is_file():
            print(f"  SKIP: file not found")
            continue

        t0 = time.time()

        try:
            import ijson
            print("  Loading with ijson streaming...")
            hands = []
            with open(path, "rb") as f:
                for hand in ijson.items(f, "item"):
                    hands.append(hand)
        except ImportError:
            print("  Loading entire file into memory...")
            with open(path) as f:
                hands = json.load(f)

        print(f"  Loaded {len(hands)} hands")

        df = extract_features(hands, src["label"], name)
        elapsed = time.time() - t0

        out_path = OUTPUT_DIR / src["output"]
        df.to_parquet(out_path, index=False)

        print(f"  Chunks: {len(df)} | Columns: {len(df.columns)} | Time: {elapsed:.1f}s")
        print(f"  Saved: {out_path}")
        print()

    print("All test features extracted.")


if __name__ == "__main__":
    main()
