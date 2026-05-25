#!/usr/bin/env python3
"""
Export a labeled Parquet table to ``source.npz`` for ``train_dann.py``.

Why this exists
---------------
DANN needs a fixed feature matrix ``X`` with shape ``[N, D]`` and labels ``y``. The trainer
also requires **the same** ``D`` and **column order** for unlabeled ``target.npz``. Writing
``feature_columns.json`` next to the npz records that contract so your target pipeline
(SSL embed + stats, DB batch jobs, etc.) can reproduce **identical** columns in the same order.

Without a pinned column list, silent mismatches (extra/missing columns, permuted order)
break domain alignment or train the wrong geometry.

Example
-------
python workspace/DANN/scripts/export_parquet_to_source_npz.py \\
  --parquet workspace/semi_supervised/ssl_embed/artifacts/ssl_embed_v1_mask_mixed/embeddings_concat/train.parquet \\
  --out-npz workspace/DANN/artifacts/source_train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parquet → source.npz + feature_columns.json")
    p.add_argument("--parquet", type=Path, required=True, help="Input .parquet path")
    p.add_argument("--out-npz", type=Path, required=True, help="Output .npz with keys X, y")
    p.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Output JSON path (default: same stem as --out-npz with .feature_columns.json)",
    )
    p.add_argument("--label-col", type=str, default="label", help="Binary label column")
    p.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns; default: all columns except label-col, parquet order",
    )
    p.add_argument(
        "--seat-bucket-col",
        type=str,
        default=None,
        help=(
            "Optional column for hybrid DANN nuisance head: writes int64 `seat_bucket` in npz "
            "(0..8 for rounded player counts 2..10). Not included in X."
        ),
    )
    return p.parse_args()


def _parse_feature_cols_arg(s: str | None) -> List[str] | None:
    if s is None or not str(s).strip():
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def _seat_bucket_np(col: pd.Series) -> np.ndarray:
    v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
    v = np.where(np.isfinite(v), np.round(v), 6.0)
    v = np.clip(v, 2.0, 10.0)
    return (v - 2.0).astype(np.int64)


def main() -> None:
    args = parse_args()
    path = args.parquet.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    df = pd.read_parquet(path)
    if args.label_col not in df.columns:
        raise SystemExit(f"Missing label column {args.label_col!r}; have {list(df.columns)}")

    explicit = _parse_feature_cols_arg(args.feature_cols)
    if explicit is not None:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise SystemExit(f"Unknown feature columns: {missing}")
        feature_cols: Sequence[str] = explicit
    else:
        feature_cols = [c for c in df.columns if c != args.label_col]

    X = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    y_raw = df[args.label_col].to_numpy()
    y = np.asarray(y_raw, dtype=np.float32).reshape(-1)
    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0.0, 1.0}) and not uniq.issubset({0, 1}):
        raise SystemExit(f"Expected binary labels in {args.label_col}; got values {sorted(uniq)[:10]}...")
    y = (y > 0.5).astype(np.float32)

    seat_bucket: np.ndarray | None = None
    seat_col_arg = args.seat_bucket_col
    if seat_col_arg is not None and str(seat_col_arg).strip():
        sc = str(seat_col_arg).strip()
        if sc not in df.columns:
            raise SystemExit(f"--seat-bucket-col {sc!r} not in parquet columns")
        seat_bucket = _seat_bucket_np(df[sc])
        if int(seat_bucket.max()) > 8 or int(seat_bucket.min()) < 0:
            raise SystemExit("internal error: seat_bucket out of 0..8 range")

    out_npz = args.out_npz.expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    if seat_bucket is not None:
        np.savez_compressed(out_npz, X=X, y=y, seat_bucket=seat_bucket)
    else:
        np.savez_compressed(out_npz, X=X, y=y)

    meta_path = args.out_meta
    if meta_path is None:
        meta_path = out_npz.with_suffix(".feature_columns.json")
    else:
        meta_path = meta_path.expanduser().resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "parquet_path": str(path),
        "label_column": args.label_col,
        "feature_columns": list(feature_cols),
        "n_rows": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "out_npz": str(out_npz),
        "seat_bucket_col": str(seat_col_arg).strip() if seat_col_arg else None,
        "n_seat_buckets": 9 if seat_bucket is not None else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_npz}  shape X={X.shape}  meta={meta_path}")


if __name__ == "__main__":
    main()
