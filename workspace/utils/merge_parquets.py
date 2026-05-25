#!/usr/bin/env python3
"""Merge two parquet files into one parquet file.


python workspace/utils/merge_parquets.py \
  --parquet-a workspace/preprocess/statistical_test/explorer/feature_3/data/test/wsop/train.parquet \
  --parquet-b workspace/preprocess/statistical_test/explorer/feature_3/data/test/wsop/val.parquet \
  --out workspace/preprocess/statistical_test/explorer/feature_3/data/test/wsop/wsop.parquet

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-a", type=Path, required=True, help="First input parquet path.")
    parser.add_argument("--parquet-b", type=Path, required=True, help="Second input parquet path.")
    parser.add_argument("--out", type=Path, required=True, help="Output merged parquet path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_a = args.parquet_a.expanduser().resolve()
    parquet_b = args.parquet_b.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not parquet_a.is_file():
        raise SystemExit(f"--parquet-a not found: {parquet_a}")
    if not parquet_b.is_file():
        raise SystemExit(f"--parquet-b not found: {parquet_b}")

    df_a = pd.read_parquet(parquet_a)
    df_b = pd.read_parquet(parquet_b)

    merged = pd.concat([df_a, df_b], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    print(f"Merged rows: {len(df_a)} + {len(df_b)} = {len(merged)}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
