#!/usr/bin/env python3
"""
ANOVA + Bonferroni + FDR + domain-shift combined report.

This script:
1) Loads labeled feature tables (train/val or explicit files),
2) Runs one-way ANOVA by label (human=0 vs bot=1) for each feature,
3) Computes Bonferroni and Benjamini-Hochberg FDR significance,
4) Optionally merges domain-shift metrics from domain_shift_probe output,
5) Writes a comprehensive CSV and PNG plots under --plots-dir (volcano, domain vs
   task signal, keep_score, task vs domain_shift scores).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


def _load_labeled_tables(
    data_dir: Optional[str],
    parquet_files: List[str],
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    if data_dir:
        d = Path(data_dir).expanduser().resolve()
        t = d / "train.parquet"
        v = d / "val.parquet"
        if not t.is_file() or not v.is_file():
            raise FileNotFoundError(f"{d}: expected train.parquet and val.parquet")
        dfs.extend([pd.read_parquet(t), pd.read_parquet(v)])
    for p in parquet_files:
        pp = Path(p).expanduser().resolve()
        if not pp.is_file():
            raise FileNotFoundError(pp)
        dfs.append(pd.read_parquet(pp))
    if not dfs:
        raise ValueError("Provide --data-dir or at least one --parquet")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if "label" not in df.columns:
        raise ValueError("No 'label' column found.")
    return df


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


def _anova_by_label(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c != "label"]
    rows: List[Dict[str, float]] = []
    for feat in feats:
        s = pd.to_numeric(df[feat], errors="coerce")
        human = s[df["label"] == 0].to_numpy(dtype=float)
        bot = s[df["label"] == 1].to_numpy(dtype=float)
        human = human[np.isfinite(human)]
        bot = bot[np.isfinite(bot)]
        if len(human) < 2 or len(bot) < 2:
            f_stat = np.nan
            p_val = np.nan
        else:
            f_stat, p_val = f_oneway(human, bot)
        rows.append(
            {
                "feature": feat,
                "n_human": int(len(human)),
                "n_bot": int(len(bot)),
                "mean_human": float(np.mean(human)) if len(human) else np.nan,
                "mean_bot": float(np.mean(bot)) if len(bot) else np.nan,
                "anova_f": float(f_stat) if np.isfinite(f_stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    p = out["p_value"].to_numpy(dtype=float)
    out["p_fdr_bh"] = _fdr_bh(p)
    bonf_alpha = 0.05 / max(1, len(out))
    out["sig_p_lt_0_05"] = out["p_value"] < 0.05
    out["sig_bonferroni"] = out["p_value"] < bonf_alpha
    out["sig_fdr_0_05"] = out["p_fdr_bh"] < 0.05
    out["bonferroni_alpha"] = bonf_alpha
    return out


def _merge_domain_shift(anova_df: pd.DataFrame, shift_csv: Optional[str]) -> pd.DataFrame:
    if not shift_csv:
        return anova_df
    p = Path(shift_csv).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    shift = pd.read_csv(p)
    cols = [c for c in ["feature", "domain_auc_macro_abs", "rf_importance", "max_abs_cohen_d_pairwise"] if c in shift.columns]
    if "feature" not in cols:
        raise ValueError(f"{p}: no 'feature' column found")
    shift = shift[cols]
    return anova_df.merge(shift, on="feature", how="left")


def _add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    task = np.log1p(out["anova_f"].clip(lower=0).fillna(0))
    task = (task - task.min()) / (task.max() - task.min() + 1e-12)
    out["task_importance_score"] = task

    if "domain_auc_macro_abs" in out.columns:
        shift = (out["domain_auc_macro_abs"].fillna(0.5) - 0.5) / 0.5
        shift = shift.clip(0, 1)
        out["domain_shift_score"] = shift
        out["keep_score"] = 0.65 * out["task_importance_score"] + 0.35 * (1 - out["domain_shift_score"])
    else:
        out["domain_shift_score"] = np.nan
        out["keep_score"] = out["task_importance_score"]
    return out


def _safe_neg_log10_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    return -np.log10(p)


def render_anova_domain_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    *,
    top_k_bars: int = 35,
    point_label_max: int = 18,
) -> List[str]:
    """
    Write PNGs under plots_dir: volcano, domain-vs-task scatter, keep_score bars.
    Domain-aware plots require columns from domain_shift merge.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Plotting requires matplotlib. Install with: python -m pip install matplotlib"
        ) from e

    plots_dir = plots_dir.expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    emitted: List[str] = []

    work = df.copy()
    work["mean_diff_bot_minus_human"] = work["mean_bot"] - work["mean_human"]
    work["neg_log10_p"] = _safe_neg_log10_p(work["p_value"].to_numpy(dtype=float))
    has_domain = "domain_auc_macro_abs" in work.columns and work["domain_auc_macro_abs"].notna().any()

    # 1) Volcano: effect vs significance, colored by domain leakage when available
    fig, ax = plt.subplots(figsize=(9, 6))
    x = work["mean_diff_bot_minus_human"].to_numpy(dtype=float)
    y = work["neg_log10_p"].to_numpy(dtype=float)
    if has_domain:
        c = work["domain_auc_macro_abs"].to_numpy(dtype=float)
        sc = ax.scatter(x, y, c=c, cmap="coolwarm", vmin=0.5, vmax=1.0, alpha=0.75, s=22, edgecolors="none")
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("domain AUC (macro |ovr|)")
    else:
        ax.scatter(x, y, c="0.35", alpha=0.6, s=22, edgecolors="none")
    ax.axhline(-np.log10(0.05), color="0.5", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_xlabel("mean(bot) − mean(human)")
    ax.set_ylabel("−log10(p-value)")
    ax.set_title("ANOVA volcano (color = domain shift when merged)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p_volcano = plots_dir / "anova_volcano.png"
    fig.savefig(p_volcano, dpi=160)
    plt.close(fig)
    emitted.append(str(p_volcano))

    # 2) Domain vs task: x = domain fingerprint strength, y = separation signal
    if has_domain:
        fig2, ax2 = plt.subplots(figsize=(8.5, 6))
        xd = work["domain_auc_macro_abs"].to_numpy(dtype=float)
        yd = work["neg_log10_p"].to_numpy(dtype=float)
        sig = work["sig_fdr_0_05"].to_numpy()
        colors = np.where(sig, "tab:orange", "0.55")
        ax2.scatter(xd, yd, c=colors, alpha=0.72, s=26, edgecolors="white", linewidths=0.35)
        ax2.axvline(0.75, color="0.45", linestyle=":", linewidth=1)
        ax2.axvline(0.9, color="0.25", linestyle=":", linewidth=1)
        ax2.set_xlabel("domain_auc_macro_abs (higher ⇒ easier to classify source)")
        ax2.set_ylabel("−log10(p-value) (ANOVA human vs bot)")
        ax2.set_title("Task signal vs domain fingerprint (orange = FDR sig)")
        ax2.set_xlim(0.48, 1.02)
        fig2.tight_layout()
        p_domain = plots_dir / "domain_vs_task_signal.png"
        fig2.savefig(p_domain, dpi=160)
        plt.close(fig2)
        emitted.append(str(p_domain))

        # 3) Composite keep_score bars
        top = work.sort_values("keep_score", ascending=False).head(max(1, top_k_bars))
        fig3, ax3 = plt.subplots(figsize=(9, max(4.0, 0.28 * len(top))))
        y_pos = np.arange(len(top))
        ax3.barh(y_pos, top["keep_score"].to_numpy(dtype=float), color="seagreen", alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top["feature"].tolist(), fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel("keep_score (higher = more task, less domain leak)")
        ax3.set_title(f"Top {len(top)} features by keep_score")
        fig3.tight_layout()
        p_keep = plots_dir / "top_keep_score.png"
        fig3.savefig(p_keep, dpi=160)
        plt.close(fig3)
        emitted.append(str(p_keep))

        # 4) domain_shift_score vs task_importance_score
        fig4, ax4 = plt.subplots(figsize=(7.5, 6))
        xs = work["domain_shift_score"].to_numpy(dtype=float)
        ys = work["task_importance_score"].to_numpy(dtype=float)
        ax4.scatter(xs, ys, c="steelblue", alpha=0.65, s=24, edgecolors="none")
        ax4.set_xlabel("domain_shift_score (0=low leak, 1=strong source separability)")
        ax4.set_ylabel("task_importance_score (normalized ANOVA F)")
        ax4.set_title("Prefer upper-left: strong label signal, weak domain fingerprint")
        fig4.tight_layout()
        p_quad = plots_dir / "task_vs_domain_shift_scores.png"
        fig4.savefig(p_quad, dpi=160)
        plt.close(fig4)
        emitted.append(str(p_quad))
    else:
        # Without domain merge: bar top by ANOVA F
        topf = work.sort_values(["anova_f", "p_value"], ascending=[False, True]).head(max(1, top_k_bars))
        fig3, ax3 = plt.subplots(figsize=(9, max(4.0, 0.28 * len(topf))))
        y_pos = np.arange(len(topf))
        ax3.barh(y_pos, topf["anova_f"].fillna(0).to_numpy(dtype=float), color="darkslateblue", alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(topf["feature"].tolist(), fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel("ANOVA F (human vs bot)")
        ax3.set_title(f"Top {len(topf)} features by F (no domain merge — run with --domain-shift-csv)")
        fig3.tight_layout()
        p_f = plots_dir / "top_anova_F.png"
        fig3.savefig(p_f, dpi=160)
        plt.close(fig3)
        emitted.append(str(p_f))

    # Optional: label a few extreme points on volcano
    if point_label_max > 0 and len(work):
        fig5, ax5 = plt.subplots(figsize=(9, 6))
        if has_domain:
            c = work["domain_auc_macro_abs"].to_numpy(dtype=float)
            ax5.scatter(x, y, c=c, cmap="coolwarm", vmin=0.5, vmax=1.0, alpha=0.55, s=18, edgecolors="none")
        else:
            ax5.scatter(x, y, c="0.35", alpha=0.55, s=18, edgecolors="none")
        ax5.axhline(-np.log10(0.05), color="0.5", linestyle="--", linewidth=1)
        abs_diff = np.abs(work["mean_diff_bot_minus_human"].to_numpy(dtype=float))
        scale = float(np.nanmax(abs_diff)) if np.any(np.isfinite(abs_diff)) else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        ranked = work.assign(_score=work["neg_log10_p"] * scale + abs_diff).sort_values("_score", ascending=False)
        pick = ranked.head(min(point_label_max, len(ranked)))
        for _, r in pick.iterrows():
            ax5.annotate(
                str(r["feature"]),
                (float(r["mean_diff_bot_minus_human"]), float(r["neg_log10_p"])),
                fontsize=7,
                alpha=0.9,
                xytext=(5, 2),
                textcoords="offset points",
            )
        ax5.set_xlabel("mean(bot) − mean(human)")
        ax5.set_ylabel("−log10(p-value)")
        ax5.set_title("Volcano with labels (top combined extremes)")
        fig5.tight_layout()
        p_lab = plots_dir / "anova_volcano_labeled.png"
        fig5.savefig(p_lab, dpi=160)
        plt.close(fig5)
        emitted.append(str(p_lab))

    return emitted


def main() -> None:
    ap = argparse.ArgumentParser(description="ANOVA + Bonferroni + FDR + optional domain-shift merge")
    ap.add_argument("--data-dir", help="Directory containing train.parquet and val.parquet")
    ap.add_argument("--parquet", action="append", default=[], help="Additional parquet file (repeatable)")
    ap.add_argument(
        "--domain-shift-csv",
        default="workspace/test/domain_shift_probe/org_bot_zenodo/feature_shift_report.csv",
        help="Optional domain shift report CSV from domain_shift_probe.py",
    )
    ap.add_argument(
        "--disable-domain-shift-merge",
        action="store_true",
        help="Skip domain shift merge even if CSV exists",
    )
    ap.add_argument(
        "--out-csv",
        default="workspace/test/anova_bonferroni_FDR_combined.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--plots-dir",
        default="workspace/preprocess/statistical_test/plots",
        help="Directory for PNG plots (volcano, domain vs task, keep_score, ...)",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip writing plots")
    ap.add_argument("--plot-top-k", type=int, default=35, help="Features in horizontal bar charts")
    ap.add_argument("--plot-label-max", type=int, default=18, help="Labeled points on volcano_labeled PNG")
    args = ap.parse_args()

    df = _load_labeled_tables(args.data_dir, args.parquet)
    out = _anova_by_label(df)
    if not args.disable_domain_shift_merge and args.domain_shift_csv:
        out = _merge_domain_shift(out, args.domain_shift_csv)
    out = _add_composite_scores(out)
    out = out.sort_values(["p_value", "anova_f"], ascending=[True, False]).reset_index(drop=True)

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    plot_paths: List[str] = []
    if not args.no_plots:
        plot_dir = Path(args.plots_dir).expanduser().resolve()
        try:
            plot_paths = render_anova_domain_plots(
                out,
                plot_dir,
                top_k_bars=args.plot_top_k,
                point_label_max=args.plot_label_max,
            )
        except RuntimeError as e:
            print(f"[anova] plots skipped: {e}")

    n_sig = int((out["sig_p_lt_0_05"] == True).sum())
    n_bonf = int((out["sig_bonferroni"] == True).sum())
    n_fdr = int((out["sig_fdr_0_05"] == True).sum())
    print(f"[anova] rows={len(out)} wrote={out_path}")
    for pp in plot_paths:
        print(f"[anova] plot: {pp}")
    print(f"[anova] significant p<0.05={n_sig} bonferroni={n_bonf} fdr={n_fdr}")
    print("[anova] top10 by p-value:")
    for _, r in out.head(10).iterrows():
        print(
            f"  {r['feature']}: p={r['p_value']:.3e}, F={r['anova_f']:.3f}, "
            f"mean_h={r['mean_human']:.6f}, mean_b={r['mean_bot']:.6f}"
        )


if __name__ == "__main__":
    main()
