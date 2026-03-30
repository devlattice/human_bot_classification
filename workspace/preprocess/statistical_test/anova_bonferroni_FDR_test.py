#!/usr/bin/env python3
"""
ANOVA + Bonferroni + FDR + domain-shift combined report.

This script:
1) Loads labeled feature tables (train/val or explicit files),
2) Runs one-way ANOVA by label (human=0 vs bot=1) for each feature,
3) Computes Bonferroni and Benjamini-Hochberg FDR significance,
4) Optionally merges domain-shift metrics from domain_shift_probe output,
5) Writes a comprehensive CSV.
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

    n_sig = int((out["sig_p_lt_0_05"] == True).sum())
    n_bonf = int((out["sig_bonferroni"] == True).sum())
    n_fdr = int((out["sig_fdr_0_05"] == True).sum())
    print(f"[anova] rows={len(out)} wrote={out_path}")
    print(f"[anova] significant p<0.05={n_sig} bonferroni={n_bonf} fdr={n_fdr}")
    print("[anova] top10 by p-value:")
    for _, r in out.head(10).iterrows():
        print(
            f"  {r['feature']}: p={r['p_value']:.3e}, F={r['anova_f']:.3f}, "
            f"mean_h={r['mean_human']:.6f}, mean_b={r['mean_bot']:.6f}"
        )


if __name__ == "__main__":
    main()
