#!/usr/bin/env python3
"""
Aggregate metrics for extra_explorer experiment logging.

Reads:
  - train_vs_validator_shift.csv (from train_validator_shift_plots.py)
  - optional cross_dataset_comparison.csv (from cross_dataset_eval.py)

Prints one JSON object to stdout. Optionally appends a row to experiments.csv.

Example:
  PYTHONPATH=. python .../collect_experiment_metrics.py \\
    --run-id E1 \\
    --shift-csv .../train_vs_validator_shift.csv \\
    --cross-eval-csv .../cross_dataset_comparison.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _summarize_shift(df: pd.DataFrame) -> Dict[str, Any]:
    ks = df["ks_statistic"].astype(float)
    sig_col = "sig_fdr_0_05"
    if sig_col in df.columns:
        sig = df[sig_col].astype(str).str.lower().isin(("true", "1", "yes"))
        n_sig = int(sig.sum())
    else:
        n_sig = None
    return {
        "n_features": int(len(df)),
        "max_ks": float(ks.max()),
        "median_ks": float(ks.median()),
        "mean_ks": float(ks.mean()),
        "count_ks_ge_0_20": int((ks >= 0.20).sum()),
        "count_ks_ge_0_15": int((ks >= 0.15).sum()),
        "n_sig_fdr_0_05": n_sig,
    }


def _pick_cross_eval_rows(df: pd.DataFrame, contains: Optional[str]) -> pd.DataFrame:
    if not contains:
        return df
    mask = df["dataset_dir"].astype(str).str.contains(contains, regex=False)
    return df.loc[mask]


def _row_metrics(sub: pd.DataFrame, path_hint: str) -> Dict[str, Any]:
    if sub.empty:
        return {
            "dataset_filter": path_hint,
            "val_roc_auc": None,
            "val_human_fpr_at_selected": None,
            "val_bot_recall_at_selected": None,
            "val_reward": None,
        }
    if len(sub) > 1:
        # Summarize: min AUC (worst case) across matched rows
        return {
            "dataset_filter": path_hint,
            "n_rows_matched": int(len(sub)),
            "val_roc_auc_min": float(sub["val_roc_auc"].min()),
            "val_roc_auc_mean": float(sub["val_roc_auc"].mean()),
            "val_human_fpr_at_selected_max": float(sub["val_human_fpr_at_selected"].max()),
            "val_bot_recall_at_selected_min": float(sub["val_bot_recall_at_selected"].min()),
            "val_reward_min": float(sub["val_reward"].min()),
        }
    r = sub.iloc[0]
    return {
        "dataset_filter": path_hint,
        "val_roc_auc": float(r["val_roc_auc"]),
        "val_human_fpr": float(r["val_human_fpr"]),
        "val_bot_recall": float(r["val_bot_recall"]),
        "val_reward": float(r["val_reward"]),
        "val_human_fpr_at_selected": float(r["val_human_fpr_at_selected"]),
        "val_bot_recall_at_selected": float(r["val_bot_recall_at_selected"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect shift + cross-eval metrics for experiment log.")
    ap.add_argument("--run-id", required=True, help="Experiment id, e.g. E1_harmonization")
    ap.add_argument("--shift-csv", type=Path, required=True, help="train_vs_validator_shift.csv")
    ap.add_argument("--cross-eval-csv", type=Path, default=None, help="cross_dataset_comparison.csv")
    ap.add_argument(
        "--cross-eval-contains",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="Repeatable: filter cross_eval rows where dataset_dir contains this substring "
        "(e.g. hollout_test.parquet). If omitted, all rows are summarized.",
    )
    ap.add_argument("--notes", default="", help="Free-text change description")
    ap.add_argument("--append-csv", type=Path, default=None, help="Append one flattened row to this CSV")
    args = ap.parse_args()

    shift_path = args.shift_csv.expanduser().resolve()
    if not shift_path.is_file():
        print(f"error: missing shift csv: {shift_path}", file=sys.stderr)
        return 1

    shift_df = pd.read_csv(shift_path)
    shift_block = _summarize_shift(shift_df)
    shift_block["shift_csv_path"] = str(shift_path)

    out: Dict[str, Any] = {
        "run_id": args.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
        "shift": shift_block,
        "cross_eval": {},
    }

    if args.cross_eval_csv:
        ce_path = args.cross_eval_csv.expanduser().resolve()
        if not ce_path.is_file():
            print(f"error: missing cross_eval csv: {ce_path}", file=sys.stderr)
            return 1
        ce = pd.read_csv(ce_path)
        out["cross_eval"]["cross_eval_csv_path"] = str(ce_path)
        filters: List[str] = list(args.cross_eval_contains) or [""]
        for hint in filters:
            sub = _pick_cross_eval_rows(ce, hint if hint else None)
            key = hint or "all_rows"
            out["cross_eval"][key] = _row_metrics(sub, key)

    print(json.dumps(out, indent=2))

    if args.append_csv:
        flat: Dict[str, Any] = {
            "run_id": out["run_id"],
            "timestamp": out["timestamp_utc"],
            "change_type": "",
            "notes": out["notes"],
            "shift_csv_path": shift_block["shift_csv_path"],
            "max_ks": shift_block["max_ks"],
            "median_ks": shift_block["median_ks"],
            "count_ks_ge_0_20": shift_block["count_ks_ge_0_20"],
            "count_ks_ge_0_15": shift_block["count_ks_ge_0_15"],
            "n_features": shift_block["n_features"],
            "n_sig_fdr_0_05": shift_block["n_sig_fdr_0_05"],
            "cross_eval_csv_path": out["cross_eval"].get("cross_eval_csv_path", ""),
        }
        # Flatten first hollout_* block if present
        for k, v in out["cross_eval"].items():
            if k in ("cross_eval_csv_path",):
                continue
            if isinstance(v, dict) and v.get("val_roc_auc") is not None:
                flat[f"ce_{k}_auc"] = v["val_roc_auc"]
                flat[f"ce_{k}_human_fpr_sel"] = v.get("val_human_fpr_at_selected")
                flat[f"ce_{k}_bot_recall_sel"] = v.get("val_bot_recall_at_selected")
                flat[f"ce_{k}_reward"] = v.get("val_reward")
            elif isinstance(v, dict) and v.get("val_roc_auc_min") is not None:
                flat[f"ce_{k}_auc_min"] = v["val_roc_auc_min"]
                flat[f"ce_{k}_human_fpr_sel_max"] = v.get("val_human_fpr_at_selected_max")
                flat[f"ce_{k}_bot_recall_sel_min"] = v.get("val_bot_recall_at_selected_min")
                flat[f"ce_{k}_reward_min"] = v.get("val_reward_min")

        dest = args.append_csv.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        row_df = pd.DataFrame([flat])
        if dest.is_file():
            row_df.to_csv(dest, mode="a", header=False, index=False)
        else:
            row_df.to_csv(dest, index=False)
        print(f"[collect_experiment_metrics] appended -> {dest}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
