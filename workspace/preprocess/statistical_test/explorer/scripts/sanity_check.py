#!/usr/bin/env python3
"""
Sanity checks for feature parquets before ANOVA / shift / training.

Per feature column (excluding label): missing rate, unique counts, finite std,
constant / near-constant flags, numeric coercion issues. Optionally reports
exact-duplicate columns and highly correlated pairs (subsampled).

Examples (repo root):

  PYTHONPATH=. python .../sanity_check.py --data-dir workspace/dataset/.../system_bot \\
    --out-csv workspace/preprocess/statistical_test/explorer/feature_3/scripts/sanity_report.csv

  PYTHONPATH=. python .../sanity_check.py \\
    --parquet path/to/train.parquet --parquet path/to/val.parquet \\
    --out-csv sanity_report.csv

  PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/scripts/sanity_check.py \
  --data-dir workspace/preprocess/statistical_test/explorer/feature_3/data/train/public_raw \
  --out-csv workspace/preprocess/statistical_test/explorer/feature_3/sanity_check/public/sanity_report.csv \
  --out-json workspace/preprocess/statistical_test/explorer/feature_3/sanity_check/public/sanity_summary.json    
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_frames(
    data_dir: Optional[Path],
    parquet_paths: Sequence[Path],
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    if data_dir is not None:
        d = data_dir.expanduser().resolve()
        for name in ("train.parquet", "val.parquet", "validator.parquet"):
            p = d / name
            if not p.is_file():
                raise FileNotFoundError(f"{d}: missing {name}")
            dfs.append(pd.read_parquet(p))
    for p in parquet_paths:
        pp = Path(p).expanduser().resolve()
        if not pp.is_file():
            raise FileNotFoundError(pp)
        dfs.append(pd.read_parquet(pp))
    if not dfs:
        raise SystemExit("Provide --data-dir or at least one --parquet")
    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def _series_effective_equal(a: pd.Series, b: pd.Series) -> bool:
    """True if elementwise equal, treating NaN == NaN."""
    if len(a) != len(b):
        return False
    same = a.eq(b) | (a.isna() & b.isna())
    return bool(same.all())


def _per_feature_report(
    df: pd.DataFrame,
    label_col: str,
    near_constant_std: float,
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}")
    feats = [c for c in df.columns if c != label_col]
    rows: List[Dict[str, Any]] = []
    n = len(df)

    for c in feats:
        raw = df[c]
        coerced = pd.to_numeric(raw, errors="coerce")
        n_nan = int(coerced.isna().sum())
        finite = coerced[np.isfinite(coerced.to_numpy(dtype=float))]
        n_fin = int(finite.size)
        if n_fin == 0:
            u_fin = 0
            std_fin = 0.0
            mn = np.nan
            mx = np.nan
        else:
            arr = finite.to_numpy(dtype=float)
            u_fin = int(np.unique(arr).size)
            std_fin = float(np.std(arr, ddof=0))
            mn = float(np.min(arr))
            mx = float(np.max(arr))

        non_numeric_mask = raw.notna() & coerced.isna()
        n_coerce_fail = int(non_numeric_mask.sum())

        is_constant = n_fin > 0 and u_fin <= 1
        is_near_const = n_fin > 0 and (is_constant or std_fin < near_constant_std)

        rows.append(
            {
                "feature": c,
                "dtype_raw": str(raw.dtype),
                "n_rows": n,
                "n_missing": n_nan,
                "pct_missing": round(100.0 * n_nan / max(1, n), 6),
                "n_finite": n_fin,
                "n_unique_finite": u_fin,
                "std_finite": std_fin,
                "min_finite": mn,
                "max_finite": mx,
                "n_non_numeric_when_present": n_coerce_fail,
                "flag_constant": is_constant,
                "flag_near_constant": bool(is_near_const),
            }
        )
    return pd.DataFrame(rows)


def _exact_duplicate_groups(df: pd.DataFrame, feature_cols: List[str]) -> List[List[str]]:
    parent: Dict[str, str] = {c: c for c in feature_cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(feature_cols)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = feature_cols[i], feature_cols[j]
            if _series_effective_equal(df[c1], df[c2]):
                union(c1, c2)

    groups: Dict[str, List[str]] = {}
    for c in feature_cols:
        r = find(c)
        groups.setdefault(r, []).append(c)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def _high_corr_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_rows: int,
    seed: int,
    corr_threshold: float,
) -> List[Tuple[str, str, float]]:
    sub = df[feature_cols]
    if max_rows > 0 and len(sub) > max_rows:
        sub = sub.sample(n=max_rows, random_state=seed)
    num = sub.apply(pd.to_numeric, errors="coerce")
    # drop cols with no variance to avoid noisy corr
    num = num.loc[:, num.std(numeric_only=True) > 0]
    cols = [c for c in num.columns if c in feature_cols]
    if len(cols) < 2:
        return []
    cmat = num[cols].corr(method="pearson", min_periods=50)
    out: List[Tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = cmat.loc[a, b]
            if np.isfinite(v) and abs(v) >= corr_threshold:
                out.append((a, b, float(v)))
    out.sort(key=lambda t: -abs(t[2]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature parquet sanity checks")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train.parquet and val.parquet",
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        dest="parquets",
        help="Labeled parquet (repeatable). Merged with --data-dir if both set.",
    )
    ap.add_argument("--label-col", default="label")
    ap.add_argument(
        "--near-constant-std",
        type=float,
        default=1e-12,
        help="flag_near_constant if finite std is below this (and not empty)",
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.999,
        help="Report pairs with |Pearson corr| >= this (subsampled)",
    )
    ap.add_argument(
        "--max-rows-corr",
        type=int,
        default=50_000,
        help="Max rows for correlation subsample (0 = use all)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Per-feature report CSV",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON with duplicate groups and high-corr pairs",
    )
    args = ap.parse_args()

    df = _load_frames(args.data_dir, args.parquets)
    label_col = args.label_col
    feats = [c for c in df.columns if c != label_col]
    if not feats:
        raise SystemExit("No feature columns (only label?)")

    report = _per_feature_report(df, label_col, args.near_constant_std)
    dup_groups = _exact_duplicate_groups(df, feats)
    corr_pairs = _high_corr_pairs(
        df,
        feats,
        max_rows=int(args.max_rows_corr),
        seed=int(args.seed),
        corr_threshold=float(args.corr_threshold),
    )

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "label_col": label_col,
        "n_flag_constant": int(report["flag_constant"].sum()),
        "n_flag_near_constant": int(report["flag_near_constant"].sum()),
        "n_with_coercion_issues": int((report["n_non_numeric_when_present"] > 0).sum()),
        "exact_duplicate_groups": dup_groups,
        "high_corr_pairs": [{"a": a, "b": b, "corr": r} for a, b, r in corr_pairs],
    }

    if args.out_json:
        jp = Path(args.out_json).expanduser().resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # stderr summary for humans
    print(f"[sanitification] rows={len(df)} features={len(feats)} wrote {out_csv}", file=sys.stderr)
    if summary["n_flag_constant"]:
        print(
            f"[sanitification] warning: {summary['n_flag_constant']} constant (finite) columns",
            file=sys.stderr,
        )
    if summary["n_flag_near_constant"]:
        print(
            f"[sanitification] note: {summary['n_flag_near_constant']} near-constant by std<{args.near_constant_std}",
            file=sys.stderr,
        )
    if summary["n_with_coercion_issues"]:
        print(
            f"[sanitification] warning: {summary['n_with_coercion_issues']} cols with non-numeric values",
            file=sys.stderr,
        )
    if dup_groups:
        print(f"[sanitification] exact duplicate groups: {dup_groups}", file=sys.stderr)
    if corr_pairs:
        top = corr_pairs[: min(8, len(corr_pairs))]
        print(f"[sanitification] high |corr| pairs (sample): {top}", file=sys.stderr)


if __name__ == "__main__":
    main()
