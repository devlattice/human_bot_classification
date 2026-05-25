#!/usr/bin/env python3
"""
Auto-generate keep/watch/drop feature lists from ANOVA output.

Primary objective:
- Drop statistically weak or pathological features.
- Keep robust, strongly separated features.
- Put borderline features into watch-list for ablation tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _as_bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v) != 0
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"1", "true", "t", "yes", "y"}
    return False


def _safe_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


def _decide(
    row: pd.Series,
    *,
    weak_task_threshold: float,
) -> Dict[str, object]:
    feat = str(row["feature"])
    p_val = _safe_float(row.get("p_value"))
    f_stat = _safe_float(row.get("anova_f"))
    sig_fdr = _as_bool(row.get("sig_fdr_0_05"))
    sig_bonf = _as_bool(row.get("sig_bonferroni"))
    task = _safe_float(row.get("task_importance_score"), 0.0)
    n_h = int(_safe_float(row.get("n_human"), 0))
    n_b = int(_safe_float(row.get("n_bot"), 0))
    mean_h = _safe_float(row.get("mean_human"))
    mean_b = _safe_float(row.get("mean_bot"))

    non_finite = (not np.isfinite(p_val)) or (not np.isfinite(f_stat))
    near_constant = np.isfinite(mean_h) and np.isfinite(mean_b) and abs(mean_h - mean_b) <= 1e-12
    degenerate = non_finite or ((n_h > 0 and n_b > 0) and near_constant and not sig_fdr)

    if degenerate:
        return {"feature": feat, "decision": "drop", "reason": "degenerate_or_constant"}
    if not sig_fdr:
        return {"feature": feat, "decision": "drop", "reason": "not_fdr_significant"}
    if (not sig_bonf) or (task < weak_task_threshold):
        return {"feature": feat, "decision": "watch", "reason": "borderline_significance_or_weak_task_score"}
    return {"feature": feat, "decision": "keep", "reason": "robust_significant"}


def _write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(items).strip()
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate keep/watch/drop feature lists from ANOVA CSV.")
    ap.add_argument("--anova-csv", required=True, help="Input ANOVA CSV.")
    ap.add_argument(
        "--out-dir",
        default="workspace/_subnet_target/preprocess/feature_selection",
        help="Output directory for generated lists and summary.",
    )
    ap.add_argument(
        "--weak-task-threshold",
        type=float,
        default=0.20,
        help="If FDR-significant but task_importance_score < threshold, mark as watch.",
    )
    args = ap.parse_args()

    anova_csv = Path(args.anova_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not anova_csv.is_file():
        raise FileNotFoundError(anova_csv)

    df = pd.read_csv(anova_csv)
    if "feature" not in df.columns:
        raise ValueError(f"{anova_csv}: missing 'feature' column")

    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        rows.append(_decide(r, weak_task_threshold=float(args.weak_task_threshold)))

    dec = pd.DataFrame(rows).drop_duplicates(subset=["feature"], keep="first")
    merged = dec.merge(df, on="feature", how="left")

    keep = sorted(merged.loc[merged["decision"] == "keep", "feature"].astype(str).tolist())
    watch = sorted(merged.loc[merged["decision"] == "watch", "feature"].astype(str).tolist())
    drop = sorted(merged.loc[merged["decision"] == "drop", "feature"].astype(str).tolist())

    _write_list(out_dir / "keep_features.txt", keep)
    _write_list(out_dir / "watch_features.txt", watch)
    _write_list(out_dir / "drop_features.txt", drop)

    summary_cols = [
        "feature",
        "decision",
        "reason",
        "p_value",
        "p_fdr_bh",
        "sig_bonferroni",
        "sig_fdr_0_05",
        "anova_f",
        "task_importance_score",
        "mean_human",
        "mean_bot",
        "n_human",
        "n_bot",
    ]
    have_cols = [c for c in summary_cols if c in merged.columns]
    merged = merged[have_cols].sort_values(
        by=["decision", "p_value", "anova_f"],
        ascending=[True, True, False],
        na_position="last",
    )
    merged.to_csv(out_dir / "selection_summary.csv", index=False)

    print(f"[select_features] input={anova_csv}")
    print(f"[select_features] output={out_dir}")
    print(f"[select_features] keep={len(keep)} watch={len(watch)} drop={len(drop)}")


if __name__ == "__main__":
    main()
