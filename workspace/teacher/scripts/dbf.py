#!/usr/bin/env python3
"""
CLI: append Distilled Behavior Features (``dbf_*``) to a parquet table.

Works on gold (labeled) or unlabeled validator rows as long as the required
tabular columns exist (same schema as feature_45_rb).

Example::

  PYTHONPATH=. python workspace/teacher/scripts/dbf.py \\
    --input-parquet workspace/feature_build/feature_selection/extended_56/data/train/feature_45_rb/train.parquet \\
    --output-parquet workspace/teacher/artifacts/dbf/train_with_dbf.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEACHER_ROOT = Path(__file__).resolve().parents[1]
if str(TEACHER_ROOT) not in sys.path:
    sys.path.insert(0, str(TEACHER_ROOT))

from dbf.features import DBF_COLUMN_NAMES, add_dbf_columns, load_dbf_quantile_bounds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append dbf_* distilled behavior columns to parquet.")
    p.add_argument("--input-parquet", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument(
        "--overwrite-existing-dbf",
        action="store_true",
        help="Replace existing dbf_* columns if present.",
    )
    p.add_argument(
        "--quantile-bounds-json",
        type=Path,
        default=None,
        help="Train-fitted winsor bounds per dbf_* column; omit for legacy per-frame quantiles.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = args.input_parquet.expanduser().resolve()
    out = args.output_parquet.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input: {inp}")
    import pandas as pd

    df = pd.read_parquet(inp)
    existing = [c for c in DBF_COLUMN_NAMES if c in df.columns]
    if existing and not args.overwrite_existing_dbf:
        raise SystemExit(f"Input already has DBF columns {existing}; use --overwrite-existing-dbf")
    if existing:
        df = df.drop(columns=list(DBF_COLUMN_NAMES), errors="ignore")
    qb_path = args.quantile_bounds_json
    quantile_bounds = None
    if qb_path is not None:
        qb_path = qb_path.expanduser().resolve()
        if not qb_path.is_file():
            raise SystemExit(f"Missing --quantile-bounds-json: {qb_path}")
        quantile_bounds = load_dbf_quantile_bounds(qb_path)
        extra = set(quantile_bounds) - set(DBF_COLUMN_NAMES)
        if extra:
            raise SystemExit(f"--quantile-bounds-json has unknown keys: {sorted(extra)}")
        dbf_columns = tuple(c for c in DBF_COLUMN_NAMES if c in quantile_bounds)
    else:
        dbf_columns = None
    df = add_dbf_columns(
        df, inplace=False, quantile_bounds=quantile_bounds, dbf_columns=dbf_columns
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    _cols = list(dbf_columns) if dbf_columns else list(DBF_COLUMN_NAMES)
    print(f"Wrote {out} rows={len(df)} cols={len(df.columns)} dbf_cols={_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
