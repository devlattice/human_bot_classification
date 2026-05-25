#!/usr/bin/env python3
"""
Train vs validator_request: per-feature distribution shift (KS) + FDR + plots.

Use for the **shift** branch of the statistical_test pipeline (no human/bot label
required on the validator file). Compares **numeric feature columns** shared by both
parquets (``label`` excluded from tests).

Outputs under ``--out-dir``:
  - ``train_vs_validator_shift.csv`` — KS statistic, p-value, BH-FDR, means/stds
  - ``summary.json`` — row counts, paths, global notes
  - PNGs: top-K KS bar chart, mean–mean scatter, optional distribution overlays

Requires: pandas, numpy, scipy, matplotlib, pyarrow (parquet).


python3 workspace/preprocess/statistical_test/train_validator_shift_plots.py \
  --train-parquet workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet \
  --validator-parquet workspace/preprocess/statistical_test/explorer/feature_2/data/validator/validator.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/feature_2/shift/public
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    q = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if not np.any(valid):
        return q
    pv = pvals[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    bh = ranked * n / (np.arange(1, n + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    out = np.empty_like(bh)
    out[order] = bh
    q[valid] = out
    return q


def _safe_neg_log10_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    return -np.log10(p)


def _sample_frame(
    df: pd.DataFrame,
    n: int,
    seed: int,
    *,
    label_col: str | None,
    stratify: bool,
) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    if stratify and label_col and label_col in df.columns:
        y = df[label_col]
        if y.notna().sum() == len(df) and y.nunique() >= 2:
            try:
                from sklearn.model_selection import train_test_split

                idx = np.arange(len(df))
                sub_idx, _ = train_test_split(
                    idx,
                    train_size=n,
                    random_state=seed,
                    stratify=y.to_numpy(),
                )
                return df.iloc[sub_idx].reset_index(drop=True)
            except ValueError:
                pass
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _shared_numeric_features(
    a: pd.DataFrame, b: pd.DataFrame, label_col: str
) -> List[str]:
    skip = {label_col}
    cols_a = set(a.columns) - skip
    cols_b = set(b.columns) - skip
    shared = sorted(cols_a & cols_b)
    out: List[str] = []
    for c in shared:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            out.append(c)
    return out


def _ks_rows(
    df_t: pd.DataFrame,
    df_v: pd.DataFrame,
    features: List[str],
    *,
    min_samples: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feat in features:
        xt = pd.to_numeric(df_t[feat], errors="coerce").to_numpy(dtype=float)
        xv = pd.to_numeric(df_v[feat], errors="coerce").to_numpy(dtype=float)
        xt = xt[np.isfinite(xt)]
        xv = xv[np.isfinite(xv)]
        if len(xt) < min_samples or len(xv) < min_samples:
            rows.append(
                {
                    "feature": feat,
                    "ks_statistic": np.nan,
                    "p_value": np.nan,
                    "n_train": int(len(xt)),
                    "n_validator": int(len(xv)),
                    "mean_train": float(np.mean(xt)) if len(xt) else np.nan,
                    "mean_validator": float(np.mean(xv)) if len(xv) else np.nan,
                    "std_train": float(np.std(xt)) if len(xt) > 1 else 0.0,
                    "std_validator": float(np.std(xv)) if len(xv) > 1 else 0.0,
                }
            )
            continue
        # Asymptotic p-values: stable for large n and avoids exact-method warnings.
        stat, p = ks_2samp(xt, xv, method="asymp")
        rows.append(
            {
                "feature": feat,
                "ks_statistic": float(stat),
                "p_value": float(p),
                "n_train": int(len(xt)),
                "n_validator": int(len(xv)),
                "mean_train": float(np.mean(xt)),
                "mean_validator": float(np.mean(xv)),
                "std_train": float(np.std(xt)) if len(xt) > 1 else 0.0,
                "std_validator": float(np.std(xv)) if len(xv) > 1 else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    p = out["p_value"].to_numpy(dtype=float)
    out["p_fdr_bh"] = _fdr_bh(p)
    out["sig_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    out["neg_log10_p"] = _safe_neg_log10_p(p)
    return out.sort_values(["ks_statistic", "p_value"], ascending=[False, True], na_position="last").reset_index(
        drop=True
    )


def _render_plots(
    report: pd.DataFrame,
    df_t: pd.DataFrame,
    df_v: pd.DataFrame,
    out_dir: Path,
    *,
    top_k: int,
    overlay_k: int,
) -> List[str]:
    import os

    # Headless / CI: avoid needing a display when saving PNG only.
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with: python -m pip install matplotlib"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    emitted: List[str] = []
    work = report[np.isfinite(report["ks_statistic"])].copy()
    if work.empty:
        print("[train_validator_shift] no finite KS stats; skipping plots", file=sys.stderr)
        return emitted

    top_k = max(1, min(top_k, len(work)))
    top = work.nlargest(top_k, "ks_statistic")

    # 1) Horizontal bar: KS statistic (primary shift readout)
    fig1, ax1 = plt.subplots(figsize=(9, max(4.0, 0.32 * len(top))))
    y_pos = np.arange(len(top))
    ax1.barh(y_pos, top["ks_statistic"].to_numpy(dtype=float), color="steelblue", alpha=0.88)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top["feature"].tolist(), fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("KS statistic (train vs validator, higher = more different)")
    ax1.set_title(f"Top {len(top)} features by two-sample KS")
    fig1.tight_layout()
    p1 = out_dir / "shift_ks_top_barh.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)
    emitted.append(str(p1))

    # 2) -log10 p with FDR highlight
    top_p = work.nlargest(top_k, "neg_log10_p")
    fig2, ax2 = plt.subplots(figsize=(9, max(4.0, 0.32 * len(top_p))))
    y2 = np.arange(len(top_p))
    sig = top_p["sig_fdr_0_05"].to_numpy()
    colors = np.where(sig, "darkorange", "0.45")
    ax2.barh(y2, top_p["neg_log10_p"].to_numpy(dtype=float), color=colors, alpha=0.88)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(top_p["feature"].tolist(), fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(-np.log10(0.05), color="0.35", linestyle="--", linewidth=1, label="p=0.05")
    ax2.set_xlabel("−log10(p-value) (KS two-sample)")
    ax2.set_title(f"Top {len(top_p)} by significance (orange = FDR q<0.05)")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    p2 = out_dir / "shift_neglog10p_top_barh.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    emitted.append(str(p2))

    # 3) Mean train vs mean validator
    fig3, ax3 = plt.subplots(figsize=(7.2, 7.0))
    mx = work["mean_train"].to_numpy(dtype=float)
    my = work["mean_validator"].to_numpy(dtype=float)
    ks = work["ks_statistic"].to_numpy(dtype=float)
    sc = ax3.scatter(mx, my, c=ks, cmap="viridis", alpha=0.65, s=28, edgecolors="white", linewidths=0.3)
    lims = [
        float(np.nanmin(np.r_[mx, my])),
        float(np.nanmax(np.r_[mx, my])),
    ]
    if lims[0] < lims[1]:
        ax3.plot(lims, lims, "k--", alpha=0.35, linewidth=1, label="y = x")
    cb = fig3.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cb.set_label("KS statistic")
    ax3.set_xlabel("Mean (train)")
    ax3.set_ylabel("Mean (validator)")
    ax3.set_title("Per-feature means: train vs validator")
    ax3.legend(loc="upper left")
    fig3.tight_layout()
    p3 = out_dir / "shift_mean_train_vs_validator.png"
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)
    emitted.append(str(p3))

    # 4) Overlaid histograms for top `overlay_k` by KS
    overlay_k = max(1, min(overlay_k, len(top)))
    feats_overlay = top["feature"].head(overlay_k).tolist()
    ncols = 3
    nrows = int(np.ceil(len(feats_overlay) / ncols))
    fig4, axes = plt.subplots(nrows, ncols, figsize=(11, 3.3 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(feats_overlay):
        ax = axes_flat[i]
        xt = pd.to_numeric(df_t[feat], errors="coerce").to_numpy(dtype=float)
        xv = pd.to_numeric(df_v[feat], errors="coerce").to_numpy(dtype=float)
        xt = xt[np.isfinite(xt)]
        xv = xv[np.isfinite(xv)]
        if len(xt) < 2 or len(xv) < 2:
            ax.set_title(feat)
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        lo = float(min(xt.min(), xv.min()))
        hi = float(max(xt.max(), xv.max()))
        if lo >= hi:
            lo, hi = lo - 1.0, hi + 1.0
        bins = np.linspace(lo, hi, 36)
        ax.hist(xt, bins=bins, alpha=0.55, color="tab:blue", density=True, label="train")
        ax.hist(xv, bins=bins, alpha=0.55, color="tab:orange", density=True, label="validator")
        ax.set_title(feat, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(len(feats_overlay), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig4.suptitle("Distribution overlay (top KS features)", fontsize=11, y=1.02)
    fig4.tight_layout()
    p4 = out_dir / "shift_distribution_overlays.png"
    fig4.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig4)
    emitted.append(str(p4))

    return emitted


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    default_train = repo / "workspace" / "dataset" / "unpreprocessed" / "train" / "train.parquet"
    default_valreq = repo / "workspace" / "ssl_data" / "raw_data" / "validator_request.parquet"
    default_out = repo / "workspace" / "preprocess" / "statistical_test" / "plots" / "train_vs_validator"

    ap = argparse.ArgumentParser(description="KS shift report + plots: labeled train vs validator_request parquet.")
    ap.add_argument("--train-parquet", type=Path, default=default_train, help="Reference / labeled train parquet")
    ap.add_argument("--validator-parquet", type=Path, default=default_valreq, help="Unlabeled validator-shaped parquet")
    ap.add_argument("--out-dir", type=Path, default=default_out, help="Output directory for CSV, JSON, PNGs")
    ap.add_argument("--label-column", default="label", help="Column excluded from shift tests (default: label)")
    ap.add_argument("--max-rows-per-source", type=int, default=50_000, help="0 = use all rows per table")
    ap.add_argument("--min-samples-per-arm", type=int, default=30, help="Minimum finite samples per side for KS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify-train-label", action="store_true", help="Stratify train sample by label when possible")
    ap.add_argument("--top-k", type=int, default=28, help="Features in bar charts")
    ap.add_argument("--overlay-k", type=int, default=6, help="Features in histogram overlay grid")
    ap.add_argument("--no-plots", action="store_true", help="Write CSV/JSON only")
    ap.add_argument(
        "--report-name",
        default="train_vs_validator_shift.csv",
        help="CSV filename inside out-dir",
    )
    args = ap.parse_args()

    train_path = args.train_parquet.expanduser().resolve()
    val_path = args.validator_parquet.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not train_path.is_file():
        print(f"[train_validator_shift] error: missing train parquet: {train_path}", file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"[train_validator_shift] error: missing validator parquet: {val_path}", file=sys.stderr)
        return 1

    print(f"[train_validator_shift] load train: {train_path}")
    df_t = pd.read_parquet(train_path)
    print(f"[train_validator_shift] load validator: {val_path}")
    df_v = pd.read_parquet(val_path)

    label_col = args.label_column
    features = _shared_numeric_features(df_t, df_v, label_col)
    if not features:
        print("[train_validator_shift] error: no shared numeric feature columns", file=sys.stderr)
        return 1

    n = args.max_rows_per_source
    df_ts = _sample_frame(df_t, n, args.seed, label_col=label_col, stratify=args.stratify_train_label)
    df_vs = _sample_frame(df_v, n, args.seed + 1, label_col=label_col, stratify=False)

    print(
        f"[train_validator_shift] rows train={len(df_ts)} (of {len(df_t)}) "
        f"validator={len(df_vs)} (of {len(df_v)}) features={len(features)}"
    )

    report = _ks_rows(df_ts, df_vs, features, min_samples=args.min_samples_per_arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.report_name
    report.to_csv(csv_path, index=False)

    n_valid = int(np.isfinite(report["ks_statistic"].to_numpy()).sum())
    n_sig = int(report["sig_fdr_0_05"].sum())
    summary = {
        "train_parquet": str(train_path),
        "validator_parquet": str(val_path),
        "n_features_tested": int(len(features)),
        "n_ks_finite": n_valid,
        "n_sig_fdr_0_05": n_sig,
        "max_rows_per_source": int(n),
        "stratify_train_label": bool(args.stratify_train_label),
        "outputs": {
            "csv": str(csv_path),
            "plots_dir": str(out_dir),
        },
        "plot_files": [],
        "plots_skip_reason": None,
        "note": "KS compares marginal distributions (train vs validator). High KS does not imply bad for human/bot task.",
    }

    plot_files: List[str] = []
    if not args.no_plots:
        try:
            plot_files = _render_plots(
                report,
                df_ts,
                df_vs,
                out_dir,
                top_k=args.top_k,
                overlay_k=args.overlay_k,
            )
            summary["plot_files"] = plot_files
        except RuntimeError as e:
            msg = str(e)
            summary["plots_skip_reason"] = msg
            print(f"[train_validator_shift] plots skipped: {msg}", file=sys.stderr)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[train_validator_shift] wrote {csv_path}")
    for p in plot_files:
        print(f"[train_validator_shift] plot: {p}")
    print(f"[train_validator_shift] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
