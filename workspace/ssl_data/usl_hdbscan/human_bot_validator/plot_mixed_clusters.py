#!/usr/bin/env python3
"""
Plots for ``mixed_train.parquet`` + ``mixed_clusters.parquet`` (same row order).

1. Heatmap: ``mix_source`` × ``cluster`` (counts).
2. Stacked bars: per ``cluster``, fraction of ``label`` 0 vs 1 (known train rows only; validator NaN dropped).

Requires matplotlib + seaborn.

  PYTHONPATH=. python .../plot_mixed_clusters.py \
    --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
    --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
    --out-dir workspace/ssl_data/usl_hdbscan/human_bot_validator

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/plot_mixed_clusters.py \
  --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --out-dir workspace/ssl_data/usl_hdbscan/human_bot_validator

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    default_dir = REPO_ROOT / "workspace" / "ssl_data" / "usl_hdbscan" / "human_bot_validator"

    ap = argparse.ArgumentParser(description="Plots for mixed train + HDBSCAN clusters.")
    ap.add_argument(
        "--mixed",
        type=Path,
        default=default_dir / "mixed_train.parquet",
        help="mixed_train.parquet (mix_source, label, features).",
    )
    ap.add_argument(
        "--clusters",
        type=Path,
        default=default_dir / "mixed_clusters.parquet",
        help="clusters.parquet from cluster_hdbscan (cluster column).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_dir,
        help="Directory for PNG outputs.",
    )
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[plot_mixed] error: pip install matplotlib seaborn", file=sys.stderr)
        return 1

    m_path = args.mixed.expanduser().resolve()
    c_path = args.clusters.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not m_path.is_file():
        print(f"[plot_mixed] error: missing --mixed {m_path}", file=sys.stderr)
        return 1
    if not c_path.is_file():
        print(f"[plot_mixed] error: missing --clusters {c_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(m_path).reset_index(drop=True)
    cl = pd.read_parquet(c_path).reset_index(drop=True)
    if len(df) != len(cl):
        print(f"[plot_mixed] error: row mismatch {len(df)} vs {len(cl)}", file=sys.stderr)
        return 1
    if "cluster" not in cl.columns:
        print("[plot_mixed] error: clusters parquet needs cluster column", file=sys.stderr)
        return 1

    df = df.copy()
    df["cluster"] = cl["cluster"].astype(int)

    if "mix_source" not in df.columns:
        print("[plot_mixed] error: mixed parquet needs mix_source", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) mix_source × cluster counts
    ct = pd.crosstab(df["mix_source"], df["cluster"])
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * ct.shape[1] + 4), max(4, 0.4 * ct.shape[0] + 2)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("mix_source × cluster (row counts)")
    fig.tight_layout()
    p1 = out_dir / "mixed_heatmap_mix_source_vs_cluster.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"[plot_mixed] wrote {p1}")

    # 2) Per cluster: label 0/1 among rows with finite label
    if "label" in df.columns:
        sub = df[pd.to_numeric(df["label"], errors="coerce").notna()].copy()
        sub["label_i"] = sub["label"].astype(int)
        # cluster × label counts
        ct2 = pd.crosstab(sub["cluster"], sub["label_i"])
        # normalize rows to fraction
        frac = ct2.div(ct2.sum(axis=1), axis=0).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(max(8, 0.35 * frac.shape[0] + 4), 5))
        frac.plot(kind="barh", stacked=True, ax=ax2, color=["#4c72b0", "#dd8452"])
        ax2.set_xlabel("fraction of rows with known label")
        ax2.set_ylabel("cluster")
        ax2.set_title("label mix (0=human, 1=bot) within cluster — train rows only")
        ax2.legend(title="label")
        fig2.tight_layout()
        p2 = out_dir / "mixed_cluster_label_fraction.png"
        fig2.savefig(p2, dpi=120)
        plt.close(fig2)
        print(f"[plot_mixed] wrote {p2}")
    else:
        print("[plot_mixed] skip label fraction plot (no label column)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
