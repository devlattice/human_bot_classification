#!/usr/bin/env python3
"""
Drop domain-fingerprint ("aggressive") features from train/val datasets.

This utility reads one dataset directory containing:
  - train.parquet
  - val.parquet

It removes selected feature columns (keeps `label`) and writes a new dataset
directory. Intended for stepwise domain-shift mitigation experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import pandas as pd


DEFAULT_TOP10: List[str] = [
    "stack_mean_max",
    "mean_pot_after_mean",
    "p6p_mean",
    "bet_ratio_mean",
    "mean_pot_after_std",
    "n_players_mean",
    "fold_ratio_std",
    "mean_pot_after_max",
    "p3_mean",
    "p3_std",
]


def _read_pair(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"{data_dir}: expected train.parquet and val.parquet")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError("Both train and val parquet must contain `label` column")
    return train_df, val_df


def _read_features_file(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    feats: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        feats.append(item)
    if not feats:
        raise ValueError(f"{path}: no features found")
    return feats


def _select_drop_list(args: argparse.Namespace) -> List[str]:
    items: List[str] = []
    if args.use_default_top10:
        items.extend(DEFAULT_TOP10)
    if args.features_file:
        items.extend(_read_features_file(Path(args.features_file).expanduser().resolve()))
    if args.drop_feature:
        items.extend(args.drop_feature)
    # Keep order, dedupe.
    seen: Set[str] = set()
    out: List[str] = []
    for f in items:
        if f not in seen:
            seen.add(f)
            out.append(f)
    if not out:
        raise ValueError(
            "No features selected to drop. Use --use-default-top10, --features-file, or --drop-feature."
        )
    return out


def _drop(df: pd.DataFrame, to_drop: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    cols = set(df.columns)
    dropped = [c for c in to_drop if c in cols and c != "label"]
    missing = [c for c in to_drop if c not in cols]
    out = df.drop(columns=dropped, errors="ignore")
    if "label" not in out.columns:
        raise RuntimeError("Internal error: label column removed")
    return out, dropped, missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop aggressive/domain-shift features from parquet dataset.")
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Input directory with train.parquet + val.parquet",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for filtered train.parquet + val.parquet",
    )
    ap.add_argument(
        "--use-default-top10",
        action="store_true",
        help="Drop default 10 features from latest domain_shift_probe report",
    )
    ap.add_argument(
        "--features-file",
        help="Text file with one feature name per line",
    )
    ap.add_argument(
        "--drop-feature",
        action="append",
        default=[],
        help="Feature to drop (repeatable)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested feature is missing",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_list = _select_drop_list(args)
    train_df, val_df = _read_pair(data_dir)

    train_out, train_dropped, train_missing = _drop(train_df, drop_list)
    val_out, val_dropped, val_missing = _drop(val_df, drop_list)

    missing_union = sorted(set(train_missing) | set(val_missing))
    dropped_union = sorted(set(train_dropped) | set(val_dropped))

    if args.strict and missing_union:
        raise ValueError(f"--strict enabled and some requested features are missing: {missing_union}")

    train_out.to_parquet(out_dir / "train.parquet", index=False)
    val_out.to_parquet(out_dir / "val.parquet", index=False)

    summary = {
        "input_data_dir": str(data_dir),
        "output_data_dir": str(out_dir),
        "requested_drop_features": drop_list,
        "dropped_features": dropped_union,
        "missing_features": missing_union,
        "train_rows": int(len(train_out)),
        "val_rows": int(len(val_out)),
        "train_features_before": int(len([c for c in train_df.columns if c != "label"])),
        "val_features_before": int(len([c for c in val_df.columns if c != "label"])),
        "train_features_after": int(len([c for c in train_out.columns if c != "label"])),
        "val_features_after": int(len([c for c in val_out.columns if c != "label"])),
    }
    (out_dir / "drop_aggressive_features_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dropped_features.txt").write_text("\n".join(dropped_union) + "\n", encoding="utf-8")

    print(f"[drop_aggressive_features] input={data_dir}")
    print(f"[drop_aggressive_features] output={out_dir}")
    print(
        "[drop_aggressive_features] features before/after "
        f"train={summary['train_features_before']}->{summary['train_features_after']} "
        f"val={summary['val_features_before']}->{summary['val_features_after']}"
    )
    if missing_union:
        print(f"[drop_aggressive_features] missing requested features: {missing_union}")
    print(f"[drop_aggressive_features] dropped: {dropped_union}")


if __name__ == "__main__":
    main()
