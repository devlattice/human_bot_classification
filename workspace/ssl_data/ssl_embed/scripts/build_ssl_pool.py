#!/usr/bin/env python3
"""
Build pooled unlabeled parquet for SSL pretraining.

The script concatenates rows from repeatable --input-parquet files and keeps a shared
feature schema. Labels (if present) are ignored for pretraining but preserved unless
--drop-non-feature-cols is enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_SKIP = {"label", "mix_source", "miner_score", "cluster", "cluster_probability", "cluster_collapsed"}


def _read_feature_list(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _infer_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DEFAULT_SKIP and pd.api.types.is_numeric_dtype(df[c])]


def _align_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns ({len(missing)}): {missing[:12]}")
    out = df.copy()
    for c in features:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Pool multiple parquet tables for SSL pretraining.")
    ap.add_argument("--input-parquet", action="append", default=[], help="Repeatable parquet path.")
    ap.add_argument("--feature-cols-file", type=Path, default=None, help="Optional fixed feature list.")
    ap.add_argument("--output-parquet", type=Path, required=True, help="Output pooled parquet path.")
    ap.add_argument(
        "--drop-non-feature-cols",
        action="store_true",
        help="Keep only feature columns in output parquet.",
    )
    args = ap.parse_args()

    in_paths = [Path(p).expanduser().resolve() for p in args.input_parquet]
    if not in_paths:
        raise SystemExit("Provide at least one --input-parquet")
    for p in in_paths:
        if not p.is_file():
            raise SystemExit(f"Missing input parquet: {p}")

    frames = [pd.read_parquet(p) for p in in_paths]
    if args.feature_cols_file is not None:
        features = _read_feature_list(args.feature_cols_file.expanduser().resolve())
        if not features:
            raise SystemExit("--feature-cols-file is empty")
    else:
        features = _infer_feature_cols(frames[0])
        if not features:
            raise SystemExit("Could not infer numeric feature columns from first parquet")

    aligned = [_align_features(df, features) for df in frames]
    pooled = pd.concat(aligned, axis=0, ignore_index=True)
    if args.drop_non_feature_cols:
        pooled = pooled[list(features)].copy()

    out_path = args.output_parquet.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(out_path, index=False)

    meta = {
        "output_parquet": str(out_path),
        "n_rows": int(len(pooled)),
        "n_features": int(len(features)),
        "feature_cols": list(features),
        "inputs": [str(p) for p in in_paths],
        "drop_non_feature_cols": bool(args.drop_non_feature_cols),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build_ssl_pool] wrote {out_path} rows={len(pooled)} features={len(features)}", file=sys.stderr)
    print(f"[build_ssl_pool] meta {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

