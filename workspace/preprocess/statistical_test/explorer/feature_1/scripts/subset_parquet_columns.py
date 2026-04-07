#!/usr/bin/env python3
"""
Drop listed feature columns from labeled parquets; keep label + all other columns.

Typical flow after sanity_check:
  1. Maintain config/drop_after_sanity.txt (one feature per line, # comments ok).
  2. Run this on --data-dir to write pruned train.parquet / val.parquet elsewhere.
  3. Re-run sanity_check on the new dir; then ANOVA + train_validator_shift on same matrix.

Examples (repo root):

  PYTHONPATH=. python .../subset_parquet_columns.py \\
    --data-dir workspace/dataset/robusted_dataset/train/system_bot \\
    --drop-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/drop_after_sanity.txt \\
    --out-dir workspace/dataset/robusted_dataset/train/system_bot_pruned \\
    --write-keep-list workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/keep_features.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd


def _read_keep_ordered(path: Path) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_drop_features(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def _subset_df(df: pd.DataFrame, label_col: str, drop: Set[str]) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing {label_col!r}")
    missing = drop - set(df.columns)
    if missing:
        print(
            f"[subset_parquet_columns] warning: drop list not in frame (ignored): "
            f"{sorted(missing)[:12]}{'...' if len(missing) > 12 else ''}",
            file=sys.stderr,
        )
    drop_eff = drop & set(df.columns)
    keep = [c for c in df.columns if c == label_col or c not in drop_eff]
    return df[keep].copy()


def _subset_df_keep(df: pd.DataFrame, label_col: str, features_ordered: List[str]) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing {label_col!r}")
    missing = [c for c in features_ordered if c not in df.columns]
    if missing:
        raise ValueError(
            f"Keep list has {len(missing)} column(s) not in parquet (showing up to 12): {missing[:12]}"
        )
    cols = [label_col] + list(features_ordered)
    return df[cols].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop or keep-listed feature columns in labeled parquets")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Input dir with train.parquet and val.parquet",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (created); writes train.parquet and val.parquet if --data-dir set",
    )
    ap.add_argument(
        "--drop-file",
        type=Path,
        default=None,
        help="One feature per line to remove (# comments ok). Mutually exclusive with --keep-file.",
    )
    ap.add_argument(
        "--keep-file",
        type=Path,
        default=None,
        help="Keep only these features (+ label), order preserved. Mutually exclusive with --drop-file.",
    )
    ap.add_argument("--label-col", default="label")
    ap.add_argument(
        "--parquet",
        nargs=2,
        action="append",
        metavar=("IN", "OUT"),
        default=[],
        dest="parquet_pairs",
        help="Extra in/out parquet pair (repeatable)",
    )
    ap.add_argument(
        "--write-keep-list",
        type=Path,
        default=None,
        help="Write remaining feature names (one per line, no label)",
    )
    args = ap.parse_args()

    if bool(args.drop_file) == bool(args.keep_file):
        raise SystemExit("Provide exactly one of --drop-file or --keep-file")

    features_keep: List[str] | None = None
    drop: Set[str] | None = None
    if args.keep_file:
        kp = Path(args.keep_file).expanduser().resolve()
        if not kp.is_file():
            raise FileNotFoundError(kp)
        features_keep = _read_keep_ordered(kp)
        if not features_keep:
            raise SystemExit("--keep-file is empty after comments/blank lines")
    else:
        drop_path = Path(args.drop_file).expanduser().resolve()
        if not drop_path.is_file():
            raise FileNotFoundError(drop_path)
        drop = _read_drop_features(drop_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_col = args.label_col
    ref_cols: List[str] | None = None

    if args.data_dir:
        d = Path(args.data_dir).expanduser().resolve()
        for name in ("train.parquet", "val.parquet"):
            inp = d / name
            if not inp.is_file():
                raise FileNotFoundError(inp)
            df = pd.read_parquet(inp)
            if features_keep is not None:
                sub = _subset_df_keep(df, label_col, features_keep)
            else:
                sub = _subset_df(df, label_col, drop or set())
            if ref_cols is None:
                ref_cols = list(sub.columns)
            elif list(sub.columns) != ref_cols:
                raise ValueError(f"{name}: column set differs from train after subset")
            outp = out_dir / name
            sub.to_parquet(outp, index=False)
            print(f"[subset_parquet_columns] {inp} -> {outp} cols={len(sub.columns)}", file=sys.stderr)

    for pair in args.parquet_pairs:
        in_p, out_p = Path(pair[0]).expanduser().resolve(), Path(pair[1]).expanduser().resolve()
        if not in_p.is_file():
            raise FileNotFoundError(in_p)
        df = pd.read_parquet(in_p)
        if features_keep is not None:
            sub = _subset_df_keep(df, label_col, features_keep)
        else:
            sub = _subset_df(df, label_col, drop or set())
        if ref_cols is None:
            ref_cols = list(sub.columns)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out_p, index=False)
        print(f"[subset_parquet_columns] {in_p} -> {out_p} cols={len(sub.columns)}", file=sys.stderr)

    if not args.data_dir and not args.parquet_pairs:
        raise SystemExit("Provide --data-dir and/or --parquet IN OUT pairs")

    if args.write_keep_list and ref_cols is not None:
        feats = [c for c in ref_cols if c != label_col]
        kp = Path(args.write_keep_list).expanduser().resolve()
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text("\n".join(feats) + "\n", encoding="utf-8")
        print(f"[subset_parquet_columns] wrote keep list {kp} n={len(feats)}", file=sys.stderr)


if __name__ == "__main__":
    main()
