"""Evaluate production retrain gates from ``retrain_summary.json``.

Uses metrics already written by ``train_production_model.py``:
  - May-8 (gold LOOCV row): ``bot_detect_rate`` → bot recall @0.5 (% of bots flagged)
  - ``unseen_test_results``: zenodo / public FPR @0.5, ACPC bot recall

Exit code 0 if all configured gates pass, 1 otherwise.

Usage:
    python workspace/hybrid/scripts/check_retrain_gates.py \\
        --summary workspace/hybrid/model_bundle/retrain_summary.json

    RETRAIN_GATES_STRICT=1 ./workspace/hybrid/retrain.sh   # fail pipeline on gate failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _may8_loocv_row(cv_results: list) -> dict | None:
    for r in cv_results:
        d = str(r.get("date", ""))
        if "05-08" in d or d.endswith("2026-05-08"):
            return r
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("workspace/hybrid/model_bundle/retrain_summary.json"),
        help="Path to retrain_summary.json from train_production_model.",
    )
    p.add_argument(
        "--min-may8-recall-pct",
        type=float,
        default=80.0,
        help="May-8 gold LOOCV: bot_detect_rate * 100 must be >= this.",
    )
    p.add_argument(
        "--max-zenodo-fpr-pct",
        type=float,
        default=2.0,
        help="Zenodo-test unseen: fpr_pct must be <= this (percent points).",
    )
    p.add_argument(
        "--max-public-fpr-pct",
        type=float,
        default=3.0,
        help="Public-test unseen: fpr_pct must be <= this (skip if no public_test).",
    )
    p.add_argument(
        "--min-acpc-recall-pct",
        type=float,
        default=90.0,
        help="ACPC bot test: recall_pct must be >= this (skip if no acpc_bot_test).",
    )
    p.add_argument(
        "--may8-date-substr",
        type=str,
        default="05-08",
        help="Pick LOOCV cv_results row whose date contains this substring.",
    )
    args = p.parse_args()

    if not args.summary.is_file():
        print(f"[gates] ERROR: summary not found: {args.summary}", file=sys.stderr)
        return 1

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    cv = data.get("cv_results") or []
    tests = data.get("unseen_test_results") or {}

    # Resolve May-8 row (prefer substring match)
    may8 = None
    for r in cv:
        if args.may8_date_substr in str(r.get("date", "")):
            may8 = r
            break
    if may8 is None:
        may8 = _may8_loocv_row(cv)

    checks: list[tuple[str, bool, str]] = []

    if may8 is None:
        checks.append(
            (
                f"may8_loocv_recall >= {args.min_may8_recall_pct:.1f}%",
                False,
                "no matching cv_results row (missing gold / date)",
            )
        )
    else:
        bdr = float(may8.get("bot_detect_rate", 0.0)) * 100.0
        ok = bdr + 1e-9 >= args.min_may8_recall_pct
        checks.append(
            (
                f"may8_loocv_recall >= {args.min_may8_recall_pct:.1f}%",
                ok,
                f"date={may8.get('date')} bot_detect@0.5={bdr:.2f}% (need >= {args.min_may8_recall_pct:.1f}%)",
            )
        )

    zt = tests.get("zenodo_test") or {}
    if "fpr_pct" in zt:
        fpr = float(zt["fpr_pct"])
        ok = fpr <= args.max_zenodo_fpr_pct + 1e-9
        checks.append(
            (
                f"zenodo_test_fpr <= {args.max_zenodo_fpr_pct:.2f}%",
                ok,
                f"fpr@0.5={fpr:.3f}% (cap {args.max_zenodo_fpr_pct:.2f}%) n={zt.get('n')}",
            )
        )

    pt = tests.get("public_test") or {}
    if "fpr_pct" in pt:
        fpr = float(pt["fpr_pct"])
        ok = fpr <= args.max_public_fpr_pct + 1e-9
        checks.append(
            (
                f"public_test_fpr <= {args.max_public_fpr_pct:.2f}%",
                ok,
                f"fpr@0.5={fpr:.3f}% (cap {args.max_public_fpr_pct:.2f}%) n={pt.get('n')}",
            )
        )

    ac = tests.get("acpc_bot_test") or {}
    if "recall_pct" in ac:
        rec = float(ac["recall_pct"])
        ok = rec + 1e-9 >= args.min_acpc_recall_pct
        checks.append(
            (
                f"acpc_bot_test_recall >= {args.min_acpc_recall_pct:.1f}%",
                ok,
                f"recall@0.5={rec:.2f}% (floor {args.min_acpc_recall_pct:.1f}%) n={ac.get('n')}",
            )
        )

    if not checks:
        print("[gates] ERROR: no gates could be evaluated (empty cv/tests)", file=sys.stderr)
        return 1

    print("")
    print("=" * 70)
    print("RETRAIN GATES (from retrain_summary.json)")
    print("=" * 70)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {detail}")
        all_ok = all_ok and ok

    print("=" * 70)
    if all_ok:
        print("OVERALL: PASS")
        print("=" * 70)
        return 0
    print("OVERALL: FAIL")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
