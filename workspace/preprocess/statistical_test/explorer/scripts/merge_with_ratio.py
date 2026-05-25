#!/usr/bin/env python3
"""Merge two parquet datasets with configurable sampling/upsampling ratios.

Example:
python workspace/preprocess/statistical_test/explorer/scripts/merge_with_ratio.py \
  --input-1 workspace/preprocess/statistical_test/explorer/feature_3/data/validator_merge/subset/validator_merge.parquet \
  --input-1-ratio 0.5 \
  --input-2 workspace/preprocess/statistical_test/explorer/feature_3/data/validator_merge/subset/validator_new.parquet \
  --input-2-ratio 10 \
  --shuffle \
  --out workspace/preprocess/statistical_test/explorer/feature_3/data/validator_merge/subset/validator_mix_2to1.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _sample_with_ratio(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    if ratio <= 0:
        raise SystemExit("ratio must be > 0")
    if ratio == 1.0:
        return df.copy()
    if ratio < 1.0:
        n = max(1, int(round(len(df) * ratio)))
        return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)

    # ratio > 1.0: upsample with replacement
    n = max(1, int(round(len(df) * ratio)))
    replace = n > len(df)
    return df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-1", type=Path, required=True, help="First input parquet (e.g., old validator).")
    p.add_argument("--input-1-ratio", type=float, required=True, help="Sampling ratio for input-1 (e.g., 0.5).")
    p.add_argument("--input-2", type=Path, required=True, help="Second input parquet (e.g., new validator).")
    p.add_argument("--input-2-ratio", type=float, required=True, help="Sampling ratio for input-2 (e.g., 10).")
    p.add_argument("--out", type=Path, required=True, help="Output merged parquet path.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle merged output rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_1 = args.input_1.expanduser().resolve()
    input_2 = args.input_2.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not input_1.is_file():
        raise SystemExit(f"--input-1 not found: {input_1}")
    if not input_2.is_file():
        raise SystemExit(f"--input-2 not found: {input_2}")

    df1 = pd.read_parquet(input_1)
    df2 = pd.read_parquet(input_2)

    # Keep only common columns if schemas differ slightly.
    common_cols = [c for c in df1.columns if c in df2.columns]
    if not common_cols:
        raise SystemExit("No common columns between input-1 and input-2.")
    if len(common_cols) != len(df1.columns) or len(common_cols) != len(df2.columns):
        print(
            f"[warn] schema mismatch; using {len(common_cols)} common columns "
            f"(input-1: {len(df1.columns)}, input-2: {len(df2.columns)})"
        )
    df1 = df1.loc[:, common_cols]
    df2 = df2.loc[:, common_cols]

    out1 = _sample_with_ratio(df1, ratio=float(args.input_1_ratio), seed=int(args.seed))
    out2 = _sample_with_ratio(df2, ratio=float(args.input_2_ratio), seed=int(args.seed) + 1)

    merged = pd.concat([out1, out2], ignore_index=True)
    if args.shuffle:
        merged = merged.sample(frac=1.0, random_state=int(args.seed) + 2).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    print(f"Input-1 rows: {len(df1)} -> sampled: {len(out1)} (ratio={args.input_1_ratio})")
    print(f"Input-2 rows: {len(df2)} -> sampled: {len(out2)} (ratio={args.input_2_ratio})")
    print(f"Merged rows: {len(merged)}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
