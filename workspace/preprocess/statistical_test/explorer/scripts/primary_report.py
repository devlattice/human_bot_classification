#!/usr/bin/env python3
"""
Merge ANOVA (task) and train-vs-validator shift (KS) tables, then assign recommendations.

Joins on ``feature``. Overlapping columns (e.g. p_value, p_fdr_bh) get suffixes
``_anova`` and ``_shift``.

Recommendations (default rules):
  - high_shift: ks_statistic >= quantile(--shift-high-quantile)
  - strong_task: sig_fdr_0_05 from ANOVA (fallback: p_value_anova < 0.05)

  keep                 — low shift, strong task
  keep_watch_shift     — high shift, strong task (stabilize / harmonize)
  drop_candidate_shift — high shift, weak task (shift without label signal)
  drop_candidate_weak  — low shift, weak task (optional prune)

Outputs under --output (dir):
  merged_primary.csv
  summary.json
  features_keep.txt
  features_keep_watch_shift.txt
  features_drop_shift_only.txt
  features_drop_weak_both.txt

Example (repo root):

  PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/primary_report.py \\
    --anova-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/task_signal/anova_bonferroni_FDR_combined.csv \\
    --shift-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/shift/train_vs_validator_shift.csv \\
    --output workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("1", "true", "yes", "t"))


def _load_csv(path: Path) -> pd.DataFrame:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _merge_tables(anova: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    if "feature" not in anova.columns or "feature" not in shift.columns:
        raise ValueError("Both CSVs need a 'feature' column")
    merged = pd.merge(
        anova,
        shift,
        on="feature",
        how="inner",
        suffixes=("_anova", "_shift"),
    )
    if merged.empty:
        raise ValueError("Merge produced zero rows (no shared feature names?)")
    return merged


def _strong_task(merged: pd.DataFrame) -> pd.Series:
    if "sig_fdr_0_05_anova" in merged.columns:
        return _read_bool_series(merged["sig_fdr_0_05_anova"])
    if "sig_fdr_0_05" in merged.columns and "sig_fdr_0_05_shift" not in merged.columns:
        return _read_bool_series(merged["sig_fdr_0_05"])
    if "p_value_anova" in merged.columns:
        return merged["p_value_anova"].astype(float) < 0.05
    if "p_value" in merged.columns:
        return merged["p_value"].astype(float) < 0.05
    raise ValueError("Cannot infer task strength: need sig_fdr_0_05_anova or p_value_anova")


def _high_shift(merged: pd.DataFrame, quantile: float) -> pd.Series:
    ks = merged["ks_statistic"].astype(float)
    thr = float(ks.quantile(quantile))
    return ks >= thr


def _recommend(high_shift: pd.Series, strong_task: pd.Series) -> pd.Series:
    out: List[str] = []
    for hs, st in zip(high_shift.tolist(), strong_task.tolist()):
        if hs and not st:
            out.append("drop_candidate_shift")
        elif hs and st:
            out.append("keep_watch_shift")
        elif not hs and st:
            out.append("keep")
        else:
            out.append("drop_candidate_weak")
    return pd.Series(out, index=high_shift.index, dtype=object)


def _write_list(path: Path, names: List[str]) -> None:
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge ANOVA + shift CSVs and write selection hints")
    ap.add_argument("--anova-csv", type=Path, required=True)
    ap.add_argument("--shift-csv", type=Path, required=True)
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (created)",
    )
    ap.add_argument(
        "--shift-high-quantile",
        type=float,
        default=0.75,
        help="Features with ks_statistic >= this quantile are 'high shift' (default: top quartile)",
    )
    args = ap.parse_args()

    q = float(args.shift_high_quantile)
    if not 0.0 < q <= 1.0:
        raise SystemExit("--shift-high-quantile must be in (0, 1]")

    anova = _load_csv(args.anova_csv)
    shift = _load_csv(args.shift_csv)
    merged = _merge_tables(anova, shift)

    strong = _strong_task(merged)
    high = _high_shift(merged, q)
    merged = merged.copy()
    merged["strong_task_fdr"] = strong
    merged["high_shift_ks_q"] = high
    merged["ks_threshold_q"] = float(merged["ks_statistic"].astype(float).quantile(q))
    merged["recommendation"] = _recommend(high, strong)

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "merged_primary.csv"
    merged.sort_values(["recommendation", "ks_statistic"], ascending=[True, False]).to_csv(
        merged_path, index=False
    )

    counts: Dict[str, Any] = {
        "n_features_merged": int(len(merged)),
        "shift_high_quantile": q,
        "ks_threshold_applied": float(merged["ks_threshold_q"].iloc[0]),
        "by_recommendation": merged["recommendation"].value_counts().to_dict(),
        "anova_csv": str(Path(args.anova_csv).resolve()),
        "shift_csv": str(Path(args.shift_csv).resolve()),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    groups = {
        "features_keep.txt": merged.loc[merged["recommendation"] == "keep", "feature"].tolist(),
        "features_keep_watch_shift.txt": merged.loc[
            merged["recommendation"] == "keep_watch_shift", "feature"
        ].tolist(),
        "features_drop_shift_only.txt": merged.loc[
            merged["recommendation"] == "drop_candidate_shift", "feature"
        ].tolist(),
        "features_drop_weak_both.txt": merged.loc[
            merged["recommendation"] == "drop_candidate_weak", "feature"
        ].tolist(),
    }
    for fname, feats in groups.items():
        _write_list(out_dir / fname, sorted(feats))

    print(f"[primary_report] merged rows={len(merged)} -> {merged_path}", file=sys.stderr)
    print(f"[primary_report] summary -> {summary_path}", file=sys.stderr)
    for k, v in counts["by_recommendation"].items():
        print(f"[primary_report]   {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
