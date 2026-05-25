#!/usr/bin/env python3
"""
Run HDBSCAN on a feature-only Parquet (e.g. ``usl_hdbscan_features.parquet``).

Do **not** name this script ``hdbscan.py`` — a file with that name in the same directory
shadows ``pip install hdbscan`` and breaks ``import hdbscan``.

Outputs a small Parquet with ``cluster`` (-1 = noise) and optional ``cluster_probability``,
same row order as the input (for merging with ``usl.parquet`` / ``plot.py``).

By default writes PNGs under ``<parent of --output>/plots`` (cluster size bar chart,
probability histogram when available). Use ``--no-plot`` to skip. Paths are listed in
the sidecar ``*.json`` meta as ``plot_paths``.

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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
    cluster_selection_epsilon: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    from hdbscan import HDBSCAN

    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=str(metric),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)
    probs: Optional[np.ndarray] = None
    if hasattr(clusterer, "probabilities_") and clusterer.probabilities_ is not None:
        probs = np.asarray(clusterer.probabilities_, dtype=np.float64)
    return labels, probs


def _train_metrics_from_clusters(
    labels: np.ndarray,
    mix_source: pd.Series,
    y_raw: pd.Series,
) -> Dict[str, Any]:
    is_train = mix_source.astype(str).isin(("train_human", "train_bot")).to_numpy()
    y_num = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    y_ok = np.isfinite(y_num) & np.isin(y_num, [0.0, 1.0])
    m = is_train & y_ok
    n_train = int(np.sum(m))
    if n_train <= 0:
        return {"n_train_labeled": 0}

    y = y_num[m].astype(np.int64)
    cl = labels[m]
    clustered = cl >= 0
    n_clustered = int(np.sum(clustered))
    coverage = float(n_clustered / n_train) if n_train > 0 else 0.0
    if n_clustered <= 0:
        return {
            "n_train_labeled": n_train,
            "n_train_clustered": 0,
            "train_clustered_coverage": coverage,
        }

    y_c = y[clustered]
    cl_c = cl[clustered]

    # Cluster -> class mapping by majority train label inside each cluster.
    cluster_to_class: Dict[int, int] = {}
    for cid in np.unique(cl_c):
        sub = y_c[cl_c == cid]
        n0 = int(np.sum(sub == 0))
        n1 = int(np.sum(sub == 1))
        cluster_to_class[int(cid)] = 1 if n1 > n0 else 0

    pred = np.array([cluster_to_class[int(c)] for c in cl_c], dtype=np.int64)
    tn = int(np.sum((y_c == 0) & (pred == 0)))
    fp = int(np.sum((y_c == 0) & (pred == 1)))
    fn = int(np.sum((y_c == 1) & (pred == 0)))
    tp = int(np.sum((y_c == 1) & (pred == 1)))
    rec0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    rec1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    bal_acc = 0.5 * (rec0 + rec1)
    acc = float((tn + tp) / len(y_c)) if len(y_c) > 0 else 0.0

    return {
        "n_train_labeled": n_train,
        "n_train_clustered": n_clustered,
        "train_clustered_coverage": coverage,
        "train_accuracy_clustered": acc,
        "train_balanced_accuracy_clustered": bal_acc,
        "confusion_tn": tn,
        "confusion_fp": fp,
        "confusion_fn": fn,
        "confusion_tp": tp,
    }


def _validator_noise(labels: np.ndarray, mix_source: pd.Series) -> Tuple[int, int, float]:
    is_val = mix_source.astype(str).to_numpy() == "validator"
    n_val = int(np.sum(is_val))
    if n_val <= 0:
        return 0, 0, float("nan")
    n_noise = int(np.sum(labels[is_val] == -1))
    return n_noise, n_val, float(n_noise / n_val)


def _write_hdbscan_plots(
    labels: np.ndarray,
    probs: Optional[np.ndarray],
    plot_dir: Path,
    stem: str,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[cluster_hdbscan] error: pip install matplotlib (or use --no-plot)", file=sys.stderr)
        raise

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(uniq)
    uniq = uniq[order]
    counts = counts[order]
    y_labels = [("noise (-1)" if int(c) == -1 else str(int(c))) for c in uniq]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(uniq) + 1)))
    ax.barh(y_labels, counts, color="#55a868")
    ax.set_xlabel("row count")
    ax.set_ylabel("cluster id")
    ax.set_title("HDBSCAN: rows per cluster (-1 = noise)")
    fig.tight_layout()
    p1 = plot_dir / f"{stem}_cluster_sizes.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(p1)

    if probs is not None and len(probs) == len(labels):
        mask = labels >= 0
        if np.any(mask):
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.hist(probs[mask], bins=40, color="#8172b3", edgecolor="white", alpha=0.9)
            ax2.set_xlabel("cluster_probability")
            ax2.set_ylabel("count (clustered rows only)")
            ax2.set_title("HDBSCAN membership strength (excludes noise)")
            fig2.tight_layout()
            p2 = plot_dir / f"{stem}_cluster_probability_hist.png"
            fig2.savefig(p2, dpi=120)
            plt.close(fig2)
            written.append(p2)

    return written


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
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for summary PNGs (default: <output-dir>/plots, i.e. parent of --output / plots).",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib summary plots.",
    )
    ap.add_argument("--grid-search", action="store_true", help="Grid-search HDBSCAN params on mixed data.")
    ap.add_argument("--grid-min-cluster-size-start", type=int, default=20)
    ap.add_argument("--grid-min-cluster-size-stop", type=int, default=100)
    ap.add_argument("--grid-min-cluster-size-step", type=int, default=10)
    ap.add_argument("--grid-min-samples-start", type=int, default=5)
    ap.add_argument("--grid-min-samples-stop", type=int, default=50)
    ap.add_argument("--grid-min-samples-step", type=int, default=10)
    ap.add_argument(
        "--min-train-balanced-accuracy",
        type=float,
        default=0.80,
        help="Feasibility gate in grid mode.",
    )
    ap.add_argument(
        "--min-train-clustered-coverage",
        type=float,
        default=0.70,
        help="Feasibility gate in grid mode.",
    )
    ap.add_argument(
        "--grid-csv",
        type=Path,
        default=None,
        help="Where to write per-grid metrics CSV (default: <output>.grid.csv).",
    )
    ap.add_argument(
        "--grid-log-every",
        type=int,
        default=1,
        help="Print grid progress every N points (default: 1).",
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
    labels: np.ndarray
    probs: Optional[np.ndarray]
    grid_summary: Dict[str, Any] = {}
    if args.grid_search:
        if "mix_source" not in df.columns or "label" not in df.columns:
            print(
                "[cluster_hdbscan] error: --grid-search requires input with mix_source and label columns.",
                file=sys.stderr,
            )
            return 1
        c_vals = list(
            range(
                int(args.grid_min_cluster_size_start),
                int(args.grid_min_cluster_size_stop),
                int(args.grid_min_cluster_size_step),
            )
        )
        s_vals = list(
            range(
                int(args.grid_min_samples_start),
                int(args.grid_min_samples_stop),
                int(args.grid_min_samples_step),
            )
        )
        if not c_vals or not s_vals:
            print("[cluster_hdbscan] error: empty grid; check --grid-* ranges.", file=sys.stderr)
            return 1
        rows: List[Dict[str, Any]] = []
        total_points = len(c_vals) * len(s_vals)
        done = 0
        t0 = time.perf_counter()
        best_seen: Optional[Dict[str, Any]] = None
        for csz in c_vals:
            for smp in s_vals:
                lab_i, prob_i = _fit_hdbscan(
                    X,
                    min_cluster_size=csz,
                    min_samples=smp,
                    metric=str(args.metric),
                    cluster_selection_epsilon=float(args.cluster_selection_epsilon),
                )
                m = _train_metrics_from_clusters(lab_i, df["mix_source"], df["label"])
                n_noise_v, n_val, frac_noise_v = _validator_noise(lab_i, df["mix_source"])
                distinct = np.unique(lab_i)
                n_cluster_ids = int(len(distinct[distinct >= 0]))
                row: Dict[str, Any] = {
                    "min_cluster_size": int(csz),
                    "min_samples": int(smp),
                    "n_distinct_clusters": n_cluster_ids,
                    "validator_noise_count": n_noise_v,
                    "validator_rows": n_val,
                    "validator_noise_frac": frac_noise_v,
                }
                row.update(m)
                feasible = (
                    float(row.get("train_balanced_accuracy_clustered", -1.0)) >= float(args.min_train_balanced_accuracy)
                    and float(row.get("train_clustered_coverage", -1.0)) >= float(args.min_train_clustered_coverage)
                )
                row["feasible"] = bool(feasible)
                rows.append(row)
                done += 1
                if feasible:
                    if best_seen is None:
                        best_seen = row
                    else:
                        a = row
                        b = best_seen
                        # Same ranking as final selection: noise asc, bal_acc desc, coverage desc, clusters asc.
                        better = False
                        if float(a["validator_noise_frac"]) < float(b["validator_noise_frac"]):
                            better = True
                        elif float(a["validator_noise_frac"]) == float(b["validator_noise_frac"]):
                            if float(a.get("train_balanced_accuracy_clustered", -1.0)) > float(
                                b.get("train_balanced_accuracy_clustered", -1.0)
                            ):
                                better = True
                            elif float(a.get("train_balanced_accuracy_clustered", -1.0)) == float(
                                b.get("train_balanced_accuracy_clustered", -1.0)
                            ):
                                if float(a.get("train_clustered_coverage", -1.0)) > float(
                                    b.get("train_clustered_coverage", -1.0)
                                ):
                                    better = True
                                elif float(a.get("train_clustered_coverage", -1.0)) == float(
                                    b.get("train_clustered_coverage", -1.0)
                                ):
                                    if int(a.get("n_distinct_clusters", 10**9)) < int(
                                        b.get("n_distinct_clusters", 10**9)
                                    ):
                                        better = True
                        if better:
                            best_seen = a
                log_every = max(1, int(args.grid_log_every))
                if done % log_every == 0 or done == total_points:
                    elapsed = time.perf_counter() - t0
                    eta = (elapsed / done) * (total_points - done) if done > 0 else float("nan")
                    msg = (
                        f"[cluster_hdbscan] grid {done}/{total_points} "
                        f"(min_cluster_size={csz}, min_samples={smp}) "
                        f"feasible={int(feasible)} "
                        f"val_noise={row.get('validator_noise_frac', float('nan')):.6f} "
                        f"train_bal_acc={row.get('train_balanced_accuracy_clustered', float('nan')):.4f} "
                        f"coverage={row.get('train_clustered_coverage', float('nan')):.4f} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                    )
                    print(msg, flush=True)
                    if best_seen is not None:
                        print(
                            "[cluster_hdbscan] grid best-so-far "
                            f"(min_cluster_size={int(best_seen['min_cluster_size'])}, "
                            f"min_samples={int(best_seen['min_samples'])}) "
                            f"val_noise={float(best_seen['validator_noise_frac']):.6f} "
                            f"train_bal_acc={float(best_seen.get('train_balanced_accuracy_clustered', float('nan'))):.4f} "
                            f"coverage={float(best_seen.get('train_clustered_coverage', float('nan'))):.4f}",
                            flush=True,
                        )
        grid_df = pd.DataFrame(rows)
        grid_csv = args.grid_csv.expanduser().resolve() if args.grid_csv is not None else args.output.expanduser().resolve().with_suffix(".grid.csv")
        grid_csv.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(grid_csv, index=False)
        feasible_df = grid_df[grid_df["feasible"] == True].copy()
        if feasible_df.empty:
            print(
                "[cluster_hdbscan] error: no feasible grid point passed train gates; see grid CSV.",
                file=sys.stderr,
            )
            print(f"[cluster_hdbscan] grid csv: {grid_csv}", file=sys.stderr)
            return 1
        feasible_df = feasible_df.sort_values(
            by=[
                "validator_noise_frac",
                "train_balanced_accuracy_clustered",
                "train_clustered_coverage",
                "n_distinct_clusters",
            ],
            ascending=[True, False, False, True],
        )
        best = feasible_df.iloc[0]
        best_c = int(best["min_cluster_size"])
        best_s = int(best["min_samples"])
        print(
            f"[cluster_hdbscan] grid best min_cluster_size={best_c} min_samples={best_s} "
            f"validator_noise_frac={best['validator_noise_frac']:.6f} "
            f"train_bal_acc={best['train_balanced_accuracy_clustered']:.4f} "
            f"train_coverage={best['train_clustered_coverage']:.4f}"
        )
        labels, probs = _fit_hdbscan(
            X,
            min_cluster_size=best_c,
            min_samples=best_s,
            metric=str(args.metric),
            cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        )
        min_samples = best_s
        args.min_cluster_size = best_c
        grid_summary = {
            "grid_csv": str(grid_csv),
            "grid_rows": int(len(grid_df)),
            "feasible_rows": int(len(feasible_df)),
            "gates": {
                "min_train_balanced_accuracy": float(args.min_train_balanced_accuracy),
                "min_train_clustered_coverage": float(args.min_train_clustered_coverage),
            },
        }
    else:
        labels, probs = _fit_hdbscan(
            X,
            min_cluster_size=int(args.min_cluster_size),
            min_samples=int(min_samples),
            metric=str(args.metric),
            cluster_selection_epsilon=float(args.cluster_selection_epsilon),
        )

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
    if args.grid_search:
        meta["grid_search"] = grid_summary
    plot_paths: list[str] = []
    if not args.no_plot:
        plot_dir = args.plot_dir.expanduser().resolve() if args.plot_dir is not None else (out_path.parent / "plots")
        stem = out_path.stem
        try:
            for p in _write_hdbscan_plots(labels, probs, plot_dir, stem):
                plot_paths.append(str(p))
                print(f"[cluster_hdbscan] plot {p}")
        except ImportError:
            return 1

    meta["plot_paths"] = plot_paths
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[cluster_hdbscan] wrote {out_path} rows={len(out)} "
        f"distinct_clusters={n_cluster_ids} noise={n_noise}"
    )
    print(f"[cluster_hdbscan] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
