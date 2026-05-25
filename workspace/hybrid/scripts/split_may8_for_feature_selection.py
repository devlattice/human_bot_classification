"""Split labeled May-8 gold into feature-selection FIT (50%) and lockbox (50%).

Reads the full May-8 hold-out (already moved out of gold train):
  workspace/hybrid/dataset/test/may8_gold_test_features.parquet

Writes:
  workspace/hybrid/dataset/train/may8_fs_train.parquet   # labeled, used in Optuna/LOOCV score
  workspace/hybrid/dataset/test/may8_fs_lockbox.parquet   # labeled, Phase 4 lockbox only

Usage:
  python workspace/hybrid/scripts/split_may8_for_feature_selection.py
  python workspace/hybrid/scripts/split_may8_for_feature_selection.py --train-frac 0.5 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
MAY8_FULL = REPO / "workspace/hybrid/dataset/test/may8_gold_test_features.parquet"
OUT_TRAIN = REPO / "workspace/hybrid/dataset/train/may8_fs_train.parquet"
OUT_LOCKBOX = REPO / "workspace/hybrid/dataset/test/may8_fs_lockbox.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=MAY8_FULL)
    ap.add_argument("--train-out", type=Path, default=OUT_TRAIN)
    ap.add_argument("--lockbox-out", type=Path, default=OUT_LOCKBOX)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[error] missing {args.input}")
        print("  Run: python workspace/hybrid/scripts/split_gold_may8_to_test.py")
        return 1

    df = pd.read_parquet(args.input)
    if "label" not in df.columns:
        print("[error] May-8 parquet needs 'label' column")
        return 1

    y = df["label"].values
    train_df, lock_df = train_test_split(
        df,
        train_size=args.train_frac,
        random_state=args.seed,
        stratify=y,
    )

    print(f"Input:   {args.input}  n={len(df)}")
    print(f"  human={(y == 0).sum()}  bot={(y == 1).sum()}")
    print(f"FS train: n={len(train_df)}  human={(train_df['label'] == 0).sum()}  bot={(train_df['label'] == 1).sum()}")
    print(f"FS lock:  n={len(lock_df)}  human={(lock_df['label'] == 0).sum()}  bot={(lock_df['label'] == 1).sum()}")

    if args.dry_run:
        print("[dry-run] no files written")
        return 0

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.lockbox_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.train_out, index=False)
    lock_df.to_parquet(args.lockbox_out, index=False)
    print(f"Saved: {args.train_out}")
    print(f"Saved: {args.lockbox_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
