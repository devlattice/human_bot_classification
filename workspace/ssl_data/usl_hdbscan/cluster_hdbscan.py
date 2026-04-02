#!/usr/bin/env python3
"""
Run HDBSCAN on a feature-only Parquet (e.g. ``usl_hdbscan_features.parquet``).

Do **not** name this script ``hdbscan.py`` — a file with that name in the same directory
shadows ``pip install hdbscan`` and breaks ``import hdbscan``.

Outputs a small Parquet with ``cluster`` (-1 = noise) and optional ``cluster_probability``,
same row order as the input (for merging with ``usl.parquet`` / ``plot.py``).

If ``build_manifest.json`` is missing, omit ``--manifest`` (or pass a bad path: a warning is printed)
and all **numeric** columns except ``label`` / ``mix_source`` / ``miner_score`` / ``cluster*`` are used.

cd /home/dr/Workspace/Poker44-subnet
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --min-cluster-size 50

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    default_features = (
        REPO_ROOT
        / "workspace"
        / "ssl_data"
        / "usl_hdbscan"
        / "data"
        / "usl_hdbscan_features.parquet"
    )
    default_out = (
        REPO_ROOT
        / "workspace"
        / "ssl_data"
        / "usl_hdbscan"
        / "data"
        / "clusters.parquet"
    )

    ap = argparse.ArgumentParser(description="HDBSCAN clustering on feature Parquet (row order preserved).")
    ap.add_argument(
        "--input",
        type=Path,
        default=default_features,
        help=f"Parquet with numeric features only (default: {default_features})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output Parquet with cluster labels (default: {default_out})",
    )
    ap.add_argument(
        "--columns",
        nargs="*",
        default=None,
        metavar="COL",
        help="Feature columns to use (default: all numeric columns except cluster/label/miner_score).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="build_manifest.json with hdbscan_feature_columns. If omitted and "
        "build_manifest.json exists beside --input, it is used automatically.",
    )
    ap.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min_cluster_size (default: 15).")
    ap.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples (default: min_cluster_size).")
    ap.add_argument(
        "--metric",
        default="euclidean",
        help="HDBSCAN metric (default: euclidean).",
    )
    ap.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon (default: 0).",
    )
    ap.add_argument(
        "--no-scale",
        action="store_true",
        help="Skip StandardScaler (features are already robust-scaled; optional extra scale).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for any randomized steps (default: 42).",
    )
    args = ap.parse_args()

    try:
        from hdbscan import HDBSCAN
    except ImportError:
        print("[cluster_hdbscan] error: pip install hdbscan", file=sys.stderr)
        return 1

    in_path = args.input.expanduser().resolve()
    if not in_path.is_file():
        print(f"[cluster_hdbscan] error: missing --input {in_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(in_path)

    manifest_path: Optional[Path] = None
    if args.manifest is not None:
        mp = args.manifest.expanduser().resolve()
        if mp.is_file():
            manifest_path = mp
        else:
            print(
                f"[cluster_hdbscan] warning: --manifest not found ({mp}); "
                "using numeric columns from --input (run build_usl_data.py to create build_manifest.json).",
                file=sys.stderr,
            )
    else:
        sidecar = in_path.parent / "build_manifest.json"
        if sidecar.is_file():
            manifest_path = sidecar
            print(f"[cluster_hdbscan] auto manifest: {manifest_path}")

    feature_cols: List[str]
    if args.columns:
        feature_cols = list(args.columns)
    elif manifest_path is not None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = payload.get("hdbscan_feature_columns")
        if not isinstance(raw, list) or not raw:
            print("[cluster_hdbscan] error: manifest missing hdbscan_feature_columns list", file=sys.stderr)
            return 1
        feature_cols = [str(x) for x in raw]
    else:
        skip = {
            "label",
            "miner_score",
            "cluster",
            "cluster_probability",
            "mix_source",
        }
        feature_cols = [
            c
            for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]
        print(f"[cluster_hdbscan] feature mode: {len(feature_cols)} numeric columns from input (no manifest).")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[cluster_hdbscan] error: missing columns: {missing[:20]}", file=sys.stderr)
        return 1

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        print(f"[cluster_hdbscan] warning: {n_nan} NaN cells → filled with 0.0", file=sys.stderr)
        X = np.nan_to_num(X, nan=0.0)

    if not args.no_scale:
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)

    np.random.seed(int(args.random_state))

    min_samples = args.min_samples if args.min_samples is not None else args.min_cluster_size

    clusterer = HDBSCAN(
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(min_samples),
        metric=str(args.metric),
        cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    probs: Optional[np.ndarray] = None
    if hasattr(clusterer, "probabilities_") and clusterer.probabilities_ is not None:
        probs = np.asarray(clusterer.probabilities_, dtype=np.float64)

    out = pd.DataFrame({"cluster": labels.astype(np.int64)})
    if probs is not None and len(probs) == len(out):
        out["cluster_probability"] = probs

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    meta_path = out_path.with_suffix(".json")
    distinct = np.unique(labels)
    n_cluster_ids = int(len(distinct[distinct >= 0]))
    n_noise = int(np.sum(labels == -1))
    meta = {
        "input_parquet": str(in_path),
        "output_parquet": str(out_path),
        "n_rows": int(len(out)),
        "n_features_used": len(feature_cols),
        "feature_columns": feature_cols,
        "manifest_used": str(manifest_path) if manifest_path is not None else None,
        "hdbscan": {
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": int(min_samples),
            "metric": str(args.metric),
            "cluster_selection_epsilon": float(args.cluster_selection_epsilon),
            "scaled": not bool(args.no_scale),
            "random_state": int(args.random_state),
        },
        "summary": {
            "n_distinct_clusters": n_cluster_ids,
            "n_noise": n_noise,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[cluster_hdbscan] wrote {out_path} rows={len(out)} "
        f"distinct_clusters={n_cluster_ids} noise={n_noise}"
    )
    print(f"[cluster_hdbscan] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
