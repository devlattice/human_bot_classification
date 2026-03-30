#!/usr/bin/env python3
"""Validate train/val eval parquet integrity before cross-dataset benchmarking."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


LEAKAGE_KEYS = {"label", "label_flag", "is_bot", "bot_family_id", "bot_version"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate eval dataset train/val integrity.")
    ap.add_argument("--data-dir", help="Directory containing train.parquet and val.parquet")
    ap.add_argument("--train", help="Explicit train parquet path")
    ap.add_argument("--val", help="Explicit val parquet path")
    ap.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow exact row overlap across train/val (not recommended).",
    )
    return ap.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.data_dir:
        d = Path(args.data_dir).expanduser().resolve()
        train = d / "train.parquet"
        val = d / "val.parquet"
    else:
        if not (args.train and args.val):
            raise ValueError("Provide --data-dir or both --train and --val")
        train = Path(args.train).expanduser().resolve()
        val = Path(args.val).expanduser().resolve()
    return train, val


def _must_exist(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _row_hashes(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    # Stable row hash from selected columns.
    view = df.loc[:, list(cols)].copy()
    text = view.to_json(orient="records", date_unit="ns")
    # Split compact JSON list into per-row hashes by using pandas index representation.
    # Simpler and deterministic for this use-case: hash tuple(row_values) row-wise.
    return view.apply(lambda r: hashlib.sha256(repr(tuple(r.values.tolist())).encode("utf-8")).hexdigest(), axis=1)


def _validate(df: pd.DataFrame, *, name: str) -> list[str]:
    issues: list[str] = []
    if "label" not in df.columns:
        issues.append(f"{name}: missing label column")
        return issues
    labels = pd.to_numeric(df["label"], errors="coerce")
    bad = labels[~labels.isin([0, 1])]
    if len(bad) > 0:
        issues.append(f"{name}: label has non-binary values (count={len(bad)})")
    n_h = int((labels == 0).sum())
    n_b = int((labels == 1).sum())
    if n_h == 0 or n_b == 0:
        issues.append(f"{name}: single-class table (human={n_h}, bot={n_b})")
    return issues


def main() -> None:
    args = _parse_args()
    train_path, val_path = _resolve_paths(args)
    _must_exist(train_path)
    _must_exist(val_path)

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    issues: list[str] = []
    issues.extend(_validate(train, name="train"))
    issues.extend(_validate(val, name="val"))

    train_feats = [c for c in train.columns if c != "label"]
    val_feats = [c for c in val.columns if c != "label"]
    if train_feats != val_feats:
        train_set, val_set = set(train_feats), set(val_feats)
        issues.append(
            "feature mismatch: "
            f"train_only={sorted(train_set - val_set)[:8]} "
            f"val_only={sorted(val_set - train_set)[:8]}"
        )

    leakage_in_features = sorted(set(train_feats) & LEAKAGE_KEYS)
    if leakage_in_features:
        issues.append(f"leakage-like feature columns present: {leakage_in_features}")

    dup_train = int(train.duplicated().sum())
    dup_val = int(val.duplicated().sum())
    if dup_train:
        issues.append(f"train has duplicate rows: {dup_train}")
    if dup_val:
        issues.append(f"val has duplicate rows: {dup_val}")

    # Exact overlap check on full row schema intersection.
    common_cols = [c for c in train.columns if c in set(val.columns)]
    if common_cols:
        h_train = set(_row_hashes(train, common_cols).tolist())
        h_val = set(_row_hashes(val, common_cols).tolist())
        overlap = len(h_train & h_val)
    else:
        overlap = 0
    if overlap > 0 and not args.allow_overlap:
        issues.append(f"exact row overlap across train/val: {overlap} rows")

    print(f"[eval-validate] train={train_path} rows={len(train)} cols={len(train.columns)}")
    print(f"[eval-validate] val={val_path} rows={len(val)} cols={len(val.columns)}")
    print(
        "[eval-validate] class balance "
        f"train(h={(train['label'] == 0).sum()}, b={(train['label'] == 1).sum()}) "
        f"val(h={(val['label'] == 0).sum()}, b={(val['label'] == 1).sum()})"
    )
    print(f"[eval-validate] train/val exact overlap={overlap}")

    if issues:
        print("[eval-validate] FAIL")
        for i, msg in enumerate(issues, start=1):
            print(f"  {i}. {msg}")
        raise SystemExit(1)

    print("[eval-validate] PASS")


if __name__ == "__main__":
    main()
