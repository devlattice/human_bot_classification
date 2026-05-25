#!/usr/bin/env python3
"""
train_qc — quality report for a ``build_total_dataset.py`` bundle.

Reads ``<bundle-dir>/manifest.json`` and ``out_train`` / ``out_val`` paths inside it,
fits row-health thresholds on **gold train rows only** (first ``n_gold`` rows),
writes ``<bundle-dir>/qc/<run_id>/`` and updates ``manifest.json`` with ``train_qc``.

Run from repo root::

  PYTHONPATH=workspace/teacher python workspace/teacher/scripts/run_total_dataset_qc.py \\
    --bundle-dir workspace/teacher/artifacts/total_dataset_with_dbf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train_qc on total_dataset bundle.")
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "teacher" / "artifacts" / "total_dataset_with_dbf",
        help="Directory containing manifest.json and train.parquet from build_total_dataset.",
    )
    p.add_argument("--run-id", type=str, default="", help="QC run id (default: UTC timestamp).")
    p.add_argument(
        "--max-gold-hard-frac",
        type=float,
        default=0.02,
        help="Gate fails if fraction of hard-tier rows on gold train exceeds this.",
    )
    p.add_argument(
        "--max-val-hard-frac",
        type=float,
        default=0.05,
        help="Gate fails if fraction of hard-tier rows on gold val exceeds this.",
    )
    p.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with code 1 when gate_result would be fail.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT / "workspace" / "teacher"))
    from train_qc.pipeline import run_train_qc_on_bundle

    out = run_train_qc_on_bundle(
        args.bundle_dir,
        run_id=args.run_id or None,
        max_gold_hard_frac=float(args.max_gold_hard_frac),
        max_val_hard_frac=float(args.max_val_hard_frac),
    )
    print(json.dumps(out, indent=2))
    if args.fail_on_gate and out.get("gate") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
