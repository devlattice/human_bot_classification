"""Move May-8 gold rows from train to test (rotation hold-out).

Reads:
  workspace/hybrid/dataset/train/gold_features.parquet

Writes:
  workspace/hybrid/dataset/test/may8_gold_test_features.parquet  (May-8 only)
  workspace/hybrid/dataset/train/gold_features.parquet           (Apr30–May7 only)

Usage:
  python workspace/hybrid/scripts/split_gold_may8_to_test.py
  python workspace/hybrid/scripts/split_gold_may8_to_test.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_TRAIN_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
MAY8_TEST_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test" / "may8_gold_test_features.parquet"
MAY8_DATE = "2026-05-08"


def split_may8(gold: pd.DataFrame, may8_date: str = MAY8_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in gold.columns:
        raise ValueError("gold_features.parquet missing 'date' column")
    dates = gold["date"].astype(str)
    may8 = gold[dates == may8_date].copy()
    train = gold[dates != may8_date].copy()
    return train, may8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-path", type=Path, default=GOLD_TRAIN_PATH)
    ap.add_argument("--may8-out", type=Path, default=MAY8_TEST_PATH)
    ap.add_argument("--may8-date", type=str, default=MAY8_DATE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.gold_path.is_file():
        print(f"[error] missing {args.gold_path}")
        return 1

    gold = pd.read_parquet(args.gold_path)
    train, may8 = split_may8(gold, args.may8_date)

    print(f"Input:  {args.gold_path}  rows={len(gold)}")
    print(f"  dates: {sorted(gold['date'].astype(str).unique())}")
    print(f"Train:  rows={len(train)}  (Apr30–May7)")
    print(f"May-8:  rows={len(may8)}  human={(may8['label'] == 0).sum()}  bot={(may8['label'] == 1).sum()}")

    if len(may8) == 0:
        print(f"[warn] no rows for date={args.may8_date!r}; nothing written")
        return 0

    if args.dry_run:
        print("[dry-run] no files written")
        return 0

    args.may8_out.parent.mkdir(parents=True, exist_ok=True)
    may8.to_parquet(args.may8_out, index=False)
    train.to_parquet(args.gold_path, index=False)

    print(f"Saved test:  {args.may8_out}")
    print(f"Saved train: {args.gold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
