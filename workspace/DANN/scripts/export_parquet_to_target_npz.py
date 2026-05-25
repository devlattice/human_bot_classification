#!/usr/bin/env python3
"""
Export an **unlabeled** Parquet table to ``target.npz`` (key ``X`` only) for ``train_dann.py``.

This is the natural counterpart to ``export_parquet_to_source_npz.py``, which also writes
``feature_columns.json`` for the **source**. For the target you must build **the same**
columns in **the same order** as the source. The recommended flow is:

1. Export source: ``export_parquet_to_source_npz.py`` → ``source_train.npz`` +
   ``source_train.feature_columns.json``.
2. Export target: this script with ``--feature-columns-json`` pointing at that JSON, so
   column order is guaranteed to match.

Do **not** reuse the **labeled** training parquet as ``target`` for real experiments: that
would duplicate source rows and **leak labels** into the domain-adaptation story. Use a
Parquet of **unlabeled** validator (or held-out) hands built with the **same** feature
pipeline and column names.

Example
-------
python workspace/DANN/scripts/export_parquet_to_target_npz.py \\
  --parquet path/to/unlabeled_hands.parquet \\
  --feature-columns-json workspace/DANN/artifacts/source_train.feature_columns.json \\
  --out-npz workspace/DANN/artifacts/target_train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parquet → target.npz (X only)")
    p.add_argument("--parquet", type=Path, required=True, help="Input .parquet path")
    p.add_argument("--out-npz", type=Path, required=True, help="Output .npz with key X")
    p.add_argument(
        "--feature-columns-json",
        type=Path,
        default=None,
        help="Path to source *.feature_columns.json (uses its feature_columns list and order)",
    )
    p.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated columns (alternative to --feature-columns-json)",
    )
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Optional JSON summary (default: out-npz with .target_export.json suffix)",
    )
    return p.parse_args()


def _parse_feature_cols_arg(s: str | None) -> List[str] | None:
    if s is None or not str(s).strip():
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def main() -> None:
    args = parse_args()
    path = args.parquet.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    if args.feature_columns_json is not None and args.feature_cols is not None:
        raise SystemExit("Use either --feature-columns-json or --feature-cols, not both")

    if args.feature_columns_json is None and args.feature_cols is None:
        raise SystemExit(
            "Provide --feature-columns-json (recommended) or --feature-cols "
            "so target columns match source."
        )

    if args.feature_columns_json is not None:
        jpath = args.feature_columns_json.expanduser().resolve()
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        feature_cols = list(meta["feature_columns"])
    else:
        feature_cols = _parse_feature_cols_arg(args.feature_cols)
        assert feature_cols is not None

    df = pd.read_parquet(path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Parquet missing columns needed for target X: {missing}")

    X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)

    out_npz = args.out_npz.expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X)

    meta_path = args.out_meta
    if meta_path is None:
        meta_path = out_npz.with_suffix(".target_export.json")
    else:
        meta_path = meta_path.expanduser().resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta = {
        "parquet_path": str(path),
        "feature_columns": feature_cols,
        "n_rows": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "out_npz": str(out_npz),
        "feature_columns_source_json": str(args.feature_columns_json)
        if args.feature_columns_json
        else None,
    }
    meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_npz}  shape X={X.shape}  meta={meta_path}")


if __name__ == "__main__":
    main()
