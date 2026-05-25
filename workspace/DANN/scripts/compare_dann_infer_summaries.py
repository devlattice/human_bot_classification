#!/usr/bin/env python3
"""
Compare two ``summary.csv`` files from ``run_dann_auto`` infer versions.

Rows are matched by ``dataset`` name, optionally normalizing ``*_with_dbf``
suffix so ``pb_1`` aligns with ``pb_1_with_dbf``.

Example::

  python workspace/DANN/scripts/compare_dann_infer_summaries.py \
    --left workspace/DANN/infer/versions/v0005/summary.csv \
    --right workspace/DANN/infer/versions/v0006/summary.csv \
    --left-label v0005 \
    --right-label v0006
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _norm_key(name: str, *, strip_dbf: bool) -> str:
    s = name.strip()
    if strip_dbf and s.endswith("_with_dbf"):
        return s[: -len("_with_dbf")]
    return s


def _load(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        ds = str(r.get("dataset", "")).strip()
        if not ds:
            continue
        out[ds] = dict(r)
    return out


def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Diff two DANN infer summary.csv files.")
    p.add_argument("--left", type=Path, required=True, help="Path to first summary.csv")
    p.add_argument("--right", type=Path, required=True, help="Path to second summary.csv")
    p.add_argument(
        "--strip-dbf-suffix",
        action="store_true",
        default=True,
        help="Match rows by stripping _with_dbf from names for join key (default: on).",
    )
    p.add_argument(
        "--no-strip-dbf-suffix",
        action="store_false",
        dest="strip_dbf_suffix",
        help="Require exact dataset string match between files.",
    )
    p.add_argument("--left-label", type=str, default="left")
    p.add_argument("--right-label", type=str, default="right")
    args = p.parse_args()

    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()
    if not left_path.is_file():
        raise SystemExit(f"Missing --left: {left_path}")
    if not right_path.is_file():
        raise SystemExit(f"Missing --right: {right_path}")

    left_raw = _load(left_path)
    right_raw = _load(right_path)

    strip = bool(args.strip_dbf_suffix)
    left_by_key: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, row in left_raw.items():
        k = _norm_key(name, strip_dbf=strip)
        left_by_key[k] = (name, row)
    right_by_key: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, row in right_raw.items():
        k = _norm_key(name, strip_dbf=strip)
        right_by_key[k] = (name, row)

    keys = sorted(set(left_by_key) | set(right_by_key))
    metric_cols = ["n", "accuracy", "roc_auc", "average_precision", "fpr", "bot_recall"]

    print(f"left:  {left_path}  ({args.left_label})")
    print(f"right: {right_path}  ({args.right_label})")
    print(f"join:  strip _with_dbf suffix = {strip}")
    print()

    hdr = (
        "key",
        "dataset_left",
        "dataset_right",
        "metric",
        args.left_label,
        args.right_label,
        "delta(right-left)",
    )
    print("\t".join(hdr))
    for k in keys:
        ln, lrow = left_by_key.get(k, ("", {}))
        rn, rrow = right_by_key.get(k, ("", {}))
        if not lrow and not rrow:
            continue
        for m in metric_cols:
            lv = _f(str(lrow.get(m, ""))) if lrow else float("nan")
            rv = _f(str(rrow.get(m, ""))) if rrow else float("nan")
            d = rv - lv if (lv == lv and rv == rv) else float("nan")

            def fmt(v: float) -> str:
                if v != v:
                    return ""
                if m == "n":
                    return str(int(v))
                return f"{v:.6g}"

            if m == "n":
                d_s = str(int(rv - lv)) if (lv == lv and rv == rv) else ""
            else:
                d_s = fmt(d) if d == d else ""
            print(
                "\t".join(
                    [
                        k,
                        ln or "-",
                        rn or "-",
                        m,
                        fmt(lv),
                        fmt(rv),
                        d_s,
                    ]
                )
            )

    left_only = sorted(k for k in keys if k in left_by_key and k not in right_by_key)
    right_only = sorted(k for k in keys if k in right_by_key and k not in left_by_key)
    if left_only:
        print()
        print(f"NOTE: only in {args.left_label} (no right row after join): {left_only}")
    if right_only:
        print()
        print(f"NOTE: only in {args.right_label} (no left row after join): {right_only}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
