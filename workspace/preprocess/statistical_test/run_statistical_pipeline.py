#!/usr/bin/env python3
"""
Run the statistical feature-screening pipeline in order (repo root as cwd).

Default steps (all on except validator build):
  1) train_validator_shift_plots.py  — KS shift vs validator_request
  2) anova_bonferroni_FDR_test.py   — human vs bot ANOVA + FDR (+ plots)
  3) select_features.py             — keep / watch / drop lists
  4) merge ANOVA CSV + shift CSV    — one table for review (pandas)

Optional first step:
  0) build_raw_dataset_for_domain.py — JSONL → validator_request.parquet

This script only orchestrates; each step remains runnable alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def _repo_root(script_path: Path) -> Path:
    # workspace/preprocess/statistical_test/run_statistical_pipeline.py → parents[3] = repo
    return script_path.resolve().parents[3]


def _run(cmd: Sequence[str], *, cwd: Path, label: str) -> None:
    print(f"\n[pipeline] === {label} ===\n[pipeline] {' '.join(cmd)}\n", flush=True)
    r = subprocess.run(list(cmd), cwd=str(cwd))
    if r.returncode != 0:
        print(f"[pipeline] FAILED ({label}) exit={r.returncode}", file=sys.stderr, flush=True)
        raise SystemExit(r.returncode)


def _merge_anova_shift(anova_csv: Path, shift_csv: Path, out_csv: Path) -> None:
    import pandas as pd

    if not anova_csv.is_file() or not shift_csv.is_file():
        print("[pipeline] skip merge: missing anova or shift CSV", flush=True)
        return
    a = pd.read_csv(anova_csv)
    s = pd.read_csv(shift_csv)
    if "feature" not in a.columns or "feature" not in s.columns:
        print("[pipeline] skip merge: no 'feature' column", flush=True)
        return
    rename = {c: f"shift_{c}" for c in s.columns if c != "feature"}
    s = s.rename(columns=rename)
    merged = a.merge(s, on="feature", how="outer")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[pipeline] merged task+shift → {out_csv}", flush=True)


def main() -> int:
    here = Path(__file__).resolve()
    default_repo = _repo_root(here)

    d_train = default_repo / "workspace" / "dataset" / "unpreprocessed" / "original_train" / "train.parquet"
    d_valreq = default_repo / "workspace" / "ssl_data" / "raw_data" / "validator_request.parquet"
    d_json = default_repo / "workspace" / "ssl_data" / "json"
    d_sample = default_repo / "workspace" / "dataset" / "unpreprocessed" / "original_train" / "train.parquet"
    d_data_dir = default_repo / "workspace" / "dataset" / "unpreprocessed" / "original_train"
    d_artifacts = default_repo / "workspace" / "preprocess" / "statistical_test" / "artifacts"
    d_plots = default_repo / "workspace" / "preprocess" / "statistical_test" / "plots"
    d_shift_plots = d_plots / "train_vs_validator"

    py = sys.executable
    build_script = default_repo / "workspace" / "ssl_data" / "build_raw_dataset_for_domain.py"
    shift_script = default_repo / "workspace" / "preprocess" / "statistical_test" / "train_validator_shift_plots.py"
    anova_script = default_repo / "workspace" / "preprocess" / "statistical_test" / "anova_bonferroni_FDR_test.py"
    select_script = default_repo / "workspace" / "preprocess" / "statistical_test" / "select_features.py"

    ap = argparse.ArgumentParser(
        description="Orchestrate shift → ANOVA → select_features → merged CSV (optional validator parquet build)."
    )
    ap.add_argument("--repo-root", type=Path, default=default_repo, help="Repository root (cwd for subprocesses)")
    ap.add_argument(
        "--build-validator-parquet",
        action="store_true",
        help="First run build_raw_dataset_for_domain.py (JSONL → validator_request.parquet)",
    )
    ap.add_argument("--skip-shift", action="store_true", help="Skip train_validator_shift_plots.py")
    ap.add_argument("--skip-anova", action="store_true", help="Skip anova_bonferroni_FDR_test.py")
    ap.add_argument("--skip-select-features", action="store_true", help="Skip select_features.py")
    ap.add_argument("--skip-merge", action="store_true", help="Skip merged ANOVA+shift CSV")
    ap.add_argument("--jsonl-dir", type=Path, default=d_json, help="For --build-validator-parquet")
    ap.add_argument("--sample-parquet", type=Path, default=d_sample, help="Schema sample for build_raw")
    ap.add_argument("--train-parquet", type=Path, default=d_train, help="Labeled train (shift + ANOVA sample ref)")
    ap.add_argument("--validator-parquet", type=Path, default=d_valreq, help="validator_request.parquet path")
    ap.add_argument(
        "--anova-data-dir",
        type=Path,
        default=d_data_dir,
        help="Directory with train.parquet + val.parquet for ANOVA",
    )
    ap.add_argument(
        "--anova-parquet",
        type=Path,
        action="append",
        default=[],
        help="Extra labeled parquet(s) for ANOVA (repeatable). If set, do not use --anova-data-dir for that run.",
    )
    ap.add_argument("--artifacts-dir", type=Path, default=d_artifacts, help="anova CSV, merge, feature_selection/")
    ap.add_argument("--plots-dir", type=Path, default=d_plots, help="ANOVA plots directory")
    ap.add_argument("--shift-out-dir", type=Path, default=d_shift_plots, help="Shift CSV + PNG output dir")
    ap.add_argument("--max-rows-per-source", type=int, default=0, help="0 = all rows (shift step)")
    ap.add_argument("--stratify-train-label", action="store_true", help="Pass to shift step")
    ap.add_argument("--shift-no-plots", action="store_true", help="train_validator_shift_plots --no-plots")
    ap.add_argument("--anova-no-plots", action="store_true", help="anova_bonferroni_FDR_test --no-plots")
    ap.add_argument(
        "--anova-domain-shift-csv",
        type=Path,
        default=None,
        help="If set, ANOVA merges this domain_shift_probe CSV (disables --disable-domain-shift-merge)",
    )
    ap.add_argument(
        "--merged-csv-name",
        default="merged_anova_and_train_vs_validator_shift.csv",
        help="Written under --artifacts-dir",
    )
    args = ap.parse_args()

    repo = args.repo_root.expanduser().resolve()
    artifacts = args.artifacts_dir.expanduser().resolve()
    anova_csv = artifacts / "anova_bonferroni_FDR_combined.csv"
    shift_csv = args.shift_out_dir.expanduser().resolve() / "train_vs_validator_shift.csv"
    merged_csv = artifacts / args.merged_csv_name
    selection_dir = artifacts / "feature_selection"

    print(f"[pipeline] repo_root={repo}", flush=True)

    if args.build_validator_parquet:
        cmd: List[str] = [
            py,
            str(build_script),
            "--input-source-dir",
            str(args.jsonl_dir.expanduser().resolve()),
            "--sample",
            str(args.sample_parquet.expanduser().resolve()),
            "--outdir",
            str(args.validator_parquet.expanduser().resolve().parent),
            "--output-name",
            args.validator_parquet.name,
        ]
        _run(cmd, cwd=repo, label="build validator_request.parquet")

    if not args.skip_shift:
        cmd = [
            py,
            str(shift_script),
            "--train-parquet",
            str(args.train_parquet.expanduser().resolve()),
            "--validator-parquet",
            str(args.validator_parquet.expanduser().resolve()),
            "--out-dir",
            str(args.shift_out_dir.expanduser().resolve()),
            "--max-rows-per-source",
            str(int(args.max_rows_per_source)),
        ]
        if args.stratify_train_label:
            cmd.append("--stratify-train-label")
        if args.shift_no_plots:
            cmd.append("--no-plots")
        _run(cmd, cwd=repo, label="train vs validator shift (KS)")

    if not args.skip_anova:
        cmd = [
            py,
            str(anova_script),
            "--out-csv",
            str(anova_csv),
            "--plots-dir",
            str(args.plots_dir.expanduser().resolve()),
        ]
        if args.anova_parquet:
            for p in args.anova_parquet:
                cmd.extend(["--parquet", str(p.expanduser().resolve())])
        else:
            cmd.extend(["--data-dir", str(args.anova_data_dir.expanduser().resolve())])
        if args.anova_domain_shift_csv is not None:
            cmd.extend(["--domain-shift-csv", str(args.anova_domain_shift_csv.expanduser().resolve())])
        else:
            cmd.append("--disable-domain-shift-merge")
        if args.anova_no_plots:
            cmd.append("--no-plots")
        _run(cmd, cwd=repo, label="ANOVA + FDR")

    if not args.skip_merge:
        try:
            _merge_anova_shift(anova_csv, shift_csv, merged_csv)
        except ImportError:
            print("[pipeline] skip merge: pandas not installed", flush=True)

    if not args.skip_select_features:
        if not anova_csv.is_file():
            print("[pipeline] skip select_features: missing ANOVA CSV", file=sys.stderr, flush=True)
        else:
            cmd = [
                py,
                str(select_script),
                "--anova-csv",
                str(anova_csv),
                "--out-dir",
                str(selection_dir),
            ]
            _run(cmd, cwd=repo, label="select_features (keep/watch/drop)")

    print("\n[pipeline] done.", flush=True)
    print(f"[pipeline] anova: {anova_csv}", flush=True)
    print(f"[pipeline] shift: {shift_csv}", flush=True)
    if merged_csv.is_file():
        print(f"[pipeline] merged: {merged_csv}", flush=True)
    print(f"[pipeline] selection: {selection_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
