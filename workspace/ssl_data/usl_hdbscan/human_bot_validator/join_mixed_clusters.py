#!/usr/bin/env python3
"""
Join ``mixed_train.parquet`` + ``mixed_clusters.parquet`` on row order (same n_rows).

``cluster_hdbscan.py`` preserves input row order; output is one wide parquet with
all train columns plus ``cluster`` / ``cluster_probability``.

  PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/join_mixed_clusters.py \
    --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
    --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
    --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    default_dir = REPO_ROOT / "workspace" / "ssl_data" / "usl_hdbscan" / "human_bot_validator"
    ap = argparse.ArgumentParser(description="Join mixed_train + mixed_clusters by row index.")
    ap.add_argument(
        "--mixed",
        type=Path,
        default=default_dir / "mixed_train.parquet",
        help="mixed_train.parquet from mix_data.py",
    )
    ap.add_argument(
        "--clusters",
        type=Path,
        default=default_dir / "mixed_clusters.parquet",
        help="Output from cluster_hdbscan.py",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=default_dir / "mixed_train_with_clusters.parquet",
        help="Wide parquet (train cols + cluster columns)",
    )
    args = ap.parse_args()

    mixed_path = args.mixed.expanduser().resolve()
    cl_path = args.clusters.expanduser().resolve()
    out_path = args.output.expanduser().resolve()

    if not mixed_path.is_file():
        print(f"[join_mixed_clusters] error: missing --mixed {mixed_path}", file=sys.stderr)
        return 1
    if not cl_path.is_file():
        print(f"[join_mixed_clusters] error: missing --clusters {cl_path}", file=sys.stderr)
        return 1

    df_m = pd.read_parquet(mixed_path)
    df_c = pd.read_parquet(cl_path)
    if len(df_m) != len(df_c):
        print(
            f"[join_mixed_clusters] error: row count mismatch mixed={len(df_m)} clusters={len(df_c)}",
            file=sys.stderr,
        )
        return 1

    overlap = set(df_m.columns) & set(df_c.columns)
    if overlap:
        print(
            f"[join_mixed_clusters] warning: dropping from clusters side (already on mixed): {sorted(overlap)}",
            file=sys.stderr,
        )
        df_c = df_c[[c for c in df_c.columns if c not in overlap]]

    out = pd.concat([df_m.reset_index(drop=True), df_c.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        f"[join_mixed_clusters] wrote {out_path} rows={len(out)} cols={len(out.columns)} "
        f"(+cluster cols: {list(df_c.columns)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
