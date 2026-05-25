#!/usr/bin/env python3
"""
Split a labeled parquet into reproducible tune/test folds.

Use this to avoid test leakage when tuning multi-objective DANN:
  - pass *_tune.parquet to --tune-parquet
  - pass *_test.parquet to --test-parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split labeled parquet into tune/test folds.")
    p.add_argument("--parquet", type=Path, required=True, help="Input labeled parquet")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--label-col", default="label", help="Binary label column")
    p.add_argument("--tune-frac", type=float, default=0.5, help="Fraction into tune fold (0,1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output file prefix (default: parquet stem)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = args.parquet.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input parquet: {inp}")
    if not (0.0 < float(args.tune_frac) < 1.0):
        raise SystemExit("--tune-frac must be in (0,1)")

    df = pd.read_parquet(inp)
    if args.label_col not in df.columns:
        raise SystemExit(f"Missing label column {args.label_col!r} in {inp}")

    y = pd.to_numeric(df[args.label_col], errors="coerce")
    if y.isna().any():
        raise SystemExit(f"Label column contains non-numeric values: {args.label_col}")
    y01 = (y > 0.5).astype(int)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else inp.stem
    tune_path = out_dir / f"{prefix}_tune.parquet"
    test_path = out_dir / f"{prefix}_test.parquet"

    # Stratified split without sklearn dependency.
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=int(args.seed)).to_numpy()
    idx0 = [i for i in rng if y01.iloc[i] == 0]
    idx1 = [i for i in rng if y01.iloc[i] == 1]

    n0_tune = int(round(len(idx0) * float(args.tune_frac)))
    n1_tune = int(round(len(idx1) * float(args.tune_frac)))

    tune_idx = idx0[:n0_tune] + idx1[:n1_tune]
    test_idx = idx0[n0_tune:] + idx1[n1_tune:]

    df_tune = df.iloc[tune_idx].sample(frac=1.0, random_state=int(args.seed) + 11).reset_index(drop=True)
    df_test = df.iloc[test_idx].sample(frac=1.0, random_state=int(args.seed) + 29).reset_index(drop=True)

    df_tune.to_parquet(tune_path, index=False)
    df_test.to_parquet(test_path, index=False)

    def _stats(d: pd.DataFrame) -> tuple[int, int]:
        yy = (pd.to_numeric(d[args.label_col], errors="coerce") > 0.5).astype(int)
        n1 = int((yy == 1).sum())
        return len(d), n1

    n_all, n1_all = _stats(df)
    n_tu, n1_tu = _stats(df_tune)
    n_te, n1_te = _stats(df_test)
    print(f"[split_labeled_parquet] input={inp} rows={n_all} bot_rate={n1_all/max(n_all,1):.4f}")
    print(f"[split_labeled_parquet] tune={tune_path} rows={n_tu} bot_rate={n1_tu/max(n_tu,1):.4f}")
    print(f"[split_labeled_parquet] test={test_path} rows={n_te} bot_rate={n1_te/max(n_te,1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

