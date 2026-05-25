#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _resolve_input(path_or_dir: Path, default_name: str) -> Path:
    p = path_or_dir.expanduser().resolve()
    if p.is_dir():
        cand = p / default_name
        if not cand.is_file():
            raise FileNotFoundError(f"{p}: missing {default_name}")
        return cand
    return p


def _check_label(path: Path) -> tuple[int, int]:
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing label column")
    return len(df), len([c for c in df.columns if c != "label"])


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare train.parquet/val.parquet for student LGBM tuning.")
    p.add_argument("--train", type=Path, required=True, help="train parquet or dir containing train.parquet")
    p.add_argument("--val", type=Path, required=True, help="val parquet or dir containing val.parquet")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    train_src = _resolve_input(args.train, "train.parquet")
    val_src = _resolve_input(args.val, "val.parquet")

    n_train, n_feat_train = _check_label(train_src)
    n_val, n_feat_val = _check_label(val_src)
    if n_feat_train != n_feat_val:
        raise ValueError(f"feature count mismatch: train={n_feat_train}, val={n_feat_val}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dst = out_dir / "train.parquet"
    val_dst = out_dir / "val.parquet"

    pd.read_parquet(train_src).to_parquet(train_dst, index=False)
    pd.read_parquet(val_src).to_parquet(val_dst, index=False)

    summary = {
        "train_src": str(train_src),
        "val_src": str(val_src),
        "out_dir": str(out_dir),
        "train_rows": int(n_train),
        "val_rows": int(n_val),
        "n_features": int(n_feat_train),
    }
    (out_dir / "student_lgbm_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

