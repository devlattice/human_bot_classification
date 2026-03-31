#!/usr/bin/env python3
"""
Build chunk-level SSL train/val Parquet from ``score_low.json`` + ``score_high.json``,
using the same aggregation as the live miner and **apply-only** robust transforms
from an existing ``transform_meta.json`` (clip / log1p / robust scale / fillna).

**Labels (default ``--label-mode source``):**
  - Rows from ``--low-score-json`` → ``label = 0`` (pseudo low bot-risk / human-like).
  - Rows from ``--high-score-json`` → ``label = 1`` (pseudo high bot-risk).

Optional ``--label-mode threshold`` sets ``label = 1`` iff ``risk_score >= --risk-threshold``
(usually redundant with split files but matches “binarize risk_score” wording).

Optional ``--refine-score-json``: path to a JSON **array** (e.g. ``refine-score-json.json``).
Each object **must** include a **``label``** field (0 = human / low bot-risk, 1 = bot / high
bot-risk) plus ``chunk`` (hand list). Without a valid binary ``label``, the row is skipped.
The same **drop → transform_meta → keep** pipeline runs as for tail SSL rows so features
match ``lgbm_keep`` inference. (Export **raw miner-view chunks**, not pre-scaled feature
vectors, unless you add a separate tool — double-applying robust transforms would be wrong.)

**Feature pipeline**
  1. ``aggregate_chunk_from_hands(chunk)`` per record (chunk-level row).
  2. Drop columns listed in ``drop_features.txt``.
  3. ``_apply_from_meta`` from ``robust_feature_transform`` (frozen stats).
  4. Keep only names in ``keep_features.txt`` plus ``label``. Column order matches
     ``robust_feature_transform`` with ``--restrict-to-keep-features``: ``label`` first,
     then keeps in file order (same as e.g. ``balanced_sources_35k_keep/train.parquet``).

  ``chunk_hash`` and teacher ``risk_score`` are read from JSON only when needed for bucketing
  or labeling; they are **not** written to Parquet (IDs add no signal; ``risk_score`` invites
  train/serve mismatch and pseudo-label leakage vs chunk features).

Outputs ``train.parquet`` / ``val.parquet`` under ``--output`` (stratified split).

Design feedback (read before trusting SSL metrics):
  - Only **low** + **high** tails are used; **medium** / uncertain chunks are dropped.
  - Labels are **pseudo** (teacher = your miner); errors reinforce themselves if you only self-train.
  - ``transform_meta`` was fit on another dataset; apply-only improves **consistency** with
    ``lgbm_keep`` inference but does not remove **domain shift** from live chunks.
  - **Feature matching:** by default ``--strict-features`` aborts if any name in
    ``keep_features.txt`` is missing after ``transform_meta``. Optional
    ``--match-columns-parquet`` checks **exact** column names and order against a reference
    ``train.parquet`` (needs pyarrow).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# Repo root = .../miner_1
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_robust_apply():
    rft_path = REPO_ROOT / "workspace" / "datasets" / "_preprocess" / "robust_feature_transform.py"
    if not rft_path.is_file():
        raise FileNotFoundError(f"robust_feature_transform.py not found: {rft_path}")
    spec = importlib.util.spec_from_file_location("robust_feature_transform", rft_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load robust_feature_transform")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._apply_from_meta


def _read_feature_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _parquet_column_names(path: Path) -> list[str]:
    """Column names from Parquet metadata only (footer / schema); does not decode row groups."""
    pa_import_err: ImportError | None = None
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except ImportError as e:
        pa_import_err = e
    except Exception:
        raise

    try:
        import fastparquet

        return list(fastparquet.ParquetFile(path).columns)
    except ImportError as e:
        raise ImportError(
            "--match-columns-parquet needs pyarrow or fastparquet (metadata-only read). "
            "Install: pip install pyarrow"
        ) from (pa_import_err or e)


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    return [x for x in data if isinstance(x, dict)]


def _rows_from_split(
    records: list[dict[str, Any]],
    *,
    label_source: int,
    label_mode: str,
    risk_threshold: float,
    aggregate_fn,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        chunk = rec.get("chunk")
        if not isinstance(chunk, list) or not chunk:
            continue
        feat = aggregate_fn(chunk)
        if not feat:
            continue
        rs = rec.get("risk_score")
        try:
            rs_f = float(rs)
        except (TypeError, ValueError):
            continue
        if label_mode == "threshold":
            y = 1 if rs_f >= risk_threshold else 0
        else:
            y = int(label_source)
        rows.append({**feat, "label": y})
    return rows


def _coerce_refine_label(raw: Any) -> int | None:
    if raw is True:
        return 1
    if raw is False:
        return 0
    if isinstance(raw, int):
        if raw == 1:
            return 1
        if raw == 0:
            return 0
        return None
    if isinstance(raw, float):
        if raw != raw:  # NaN
            return None
        if raw == 1.0:
            return 1
        if raw == 0.0:
            return 0
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true", "yes", "bot"):
            return 1
        if s in ("0", "false", "no", "human"):
            return 0
    return None


def _rows_from_refine(
    records: list[dict[str, Any]],
    *,
    aggregate_fn,
) -> list[dict[str, Any]]:
    """Refined chunks: each object must include JSON field ``label`` (0 or 1)."""
    rows: list[dict[str, Any]] = []
    skipped_no_label = 0
    for rec in records:
        chunk = rec.get("chunk")
        if not isinstance(chunk, list) or not chunk:
            continue
        y = _coerce_refine_label(rec.get("label"))
        if y is None:
            skipped_no_label += 1
            continue
        feat = aggregate_fn(chunk)
        if not feat:
            continue
        rows.append({**feat, "label": y})
    if skipped_no_label and records:
        print(
            f"[build_dataset] refine: skipped {skipped_no_label} record(s) with missing or invalid "
            "`label` (require 0 or 1).",
            file=sys.stderr,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build robust SSL chunk-level train/val parquet from score_low / score_high JSON."
    )
    parser.add_argument(
        "--low-score-json",
        type=Path,
        required=True,
        help="JSON array of {chunk, chunk_hash, risk_score} (low bucket).",
    )
    parser.add_argument(
        "--high-score-json",
        type=Path,
        required=True,
        help="JSON array (high bucket).",
    )
    parser.add_argument(
        "--refine-score-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON array file (e.g. refine-score-json.json). Every object must have "
            "`label` (0 or 1) and `chunk`; optional `chunk_hash`, `risk_score`. Same pipeline "
            "as tail SSL. Rows without a valid binary label are skipped."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        type=Path,
        required=True,
        help="Directory containing keep_features.txt and drop_features.txt",
    )
    parser.add_argument(
        "--transform-meta",
        type=Path,
        required=True,
        help="transform_meta.json from robust preprocessing (apply-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (train.parquet, val.parquet, build_manifest.json).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction (default 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("source", "threshold"),
        default="source",
        help="source: label from file (low=0, high=1). threshold: label from risk_score vs --risk-threshold.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Used when --label-mode threshold (default 0.5).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write train.csv / val.csv instead of parquet (use if pyarrow is not installed).",
    )
    parser.add_argument(
        "--strict-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every keep_features.txt name to exist after transform (default: true). "
            "Use --no-strict-features to only warn when some are missing."
        ),
    )
    parser.add_argument(
        "--match-columns-parquet",
        type=Path,
        default=None,
        help=(
            "Reference Parquet (e.g. robust train.parquet). With --strict-features, column list "
            "is checked against keep_features.txt before loading score JSON (metadata-only read; "
            "pyarrow or fastparquet). Otherwise checked after the build."
        ),
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from poker44.validator.chunk_features import aggregate_chunk_from_hands

    apply_from_meta = _load_robust_apply()

    low_path = args.low_score_json.expanduser().resolve()
    high_path = args.high_score_json.expanduser().resolve()
    fs_dir = args.feature_selection.expanduser().resolve()
    meta_path = args.transform_meta.expanduser().resolve()
    out_dir = args.output.expanduser().resolve()

    keep_path = fs_dir / "keep_features.txt"
    drop_path = fs_dir / "drop_features.txt"
    keep_list = _read_feature_list(keep_path)
    drop_list = _read_feature_list(drop_path)

    ref_path: Path | None = None
    ref_columns_expected: list[str] | None = None
    if args.match_columns_parquet is not None:
        ref_path = args.match_columns_parquet.expanduser().resolve()
        ref_columns_expected = _parquet_column_names(ref_path)
        if args.strict_features:
            want_cols = ["label"] + keep_list
            if ref_columns_expected != want_cols:
                print(
                    "Error: keep_features.txt column set/order does not match "
                    "--match-columns-parquet (checked before loading score JSON).\n"
                    f"  reference ({ref_path}): {ref_columns_expected}\n"
                    f"  expected ['label'] + keep_features.txt: {want_cols}",
                    file=sys.stderr,
                )
                return 1

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    low_recs = _load_json_array(low_path)
    high_recs = _load_json_array(high_path)
    refine_path: Path | None = None
    refine_recs: list[dict[str, Any]] = []
    if args.refine_score_json is not None:
        refine_path = args.refine_score_json.expanduser().resolve()
        refine_recs = _load_json_array(refine_path)

    rows: list[dict[str, Any]] = []
    rows.extend(
        _rows_from_split(
            low_recs,
            label_source=0,
            label_mode=args.label_mode,
            risk_threshold=float(args.risk_threshold),
            aggregate_fn=aggregate_chunk_from_hands,
        )
    )
    rows.extend(
        _rows_from_split(
            high_recs,
            label_source=1,
            label_mode=args.label_mode,
            risk_threshold=float(args.risk_threshold),
            aggregate_fn=aggregate_chunk_from_hands,
        )
    )
    if refine_recs:
        rows.extend(
            _rows_from_refine(
                refine_recs,
                aggregate_fn=aggregate_chunk_from_hands,
            )
        )

    if not rows:
        print(
            "Error: no valid rows (check chunk / risk_score for tails; chunk / label for refine).",
            file=sys.stderr,
        )
        return 1

    df = pd.DataFrame(rows)
    n_before_drop = len(df)

    to_drop = [c for c in drop_list if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    n_after_feature_drop = len(df)

    df_t = apply_from_meta(df, meta)

    keep_present = [c for c in keep_list if c in df_t.columns]
    missing_keep = [c for c in keep_list if c not in df_t.columns]
    if missing_keep:
        msg = (
            f"keep_features missing after transform ({len(missing_keep)}): "
            f"{missing_keep[:24]}{'...' if len(missing_keep) > 24 else ''}"
        )
        if args.strict_features:
            print(f"Error: {msg}", file=sys.stderr)
            return 1
        print(f"Warning: {msg}", file=sys.stderr)

    # Same column order as robust_feature_transform.restrict_to_keep_features: label, then keeps.
    out_cols: list[str] = ["label"] + keep_present
    df_final = df_t[out_cols]

    if ref_columns_expected is not None and not args.strict_features:
        got = list(df_final.columns)
        if ref_columns_expected != got:
            print(
                "Error: output columns do not match --match-columns-parquet.\n"
                f"  reference ({ref_path}): {ref_columns_expected}\n"
                f"  built:              {got}",
                file=sys.stderr,
            )
            return 1

    vf = float(args.val_fraction)
    if not (0.0 < vf < 1.0):
        print("Error: --val-fraction must be in (0, 1).", file=sys.stderr)
        return 1

    strat = df_final["label"] if df_final["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df_final,
        test_size=vf,
        random_state=int(args.random_state),
        stratify=strat,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv:
        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
    else:
        train_path = out_dir / "train.parquet"
        val_path = out_dir / "val.parquet"
        try:
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
        except ImportError as e:
            print(
                "Error: parquet engine missing (install pyarrow: pip install pyarrow). "
                "Or re-run with --csv.",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

    manifest = {
        "low_json": str(low_path),
        "high_json": str(high_path),
        "refine_json": str(refine_path) if refine_path else None,
        "n_low_records_file": len(low_recs),
        "n_high_records_file": len(high_recs),
        "n_refine_records_file": len(refine_recs),
        "n_rows_built": n_before_drop,
        "n_rows_after_drop_features": n_after_feature_drop,
        "dropped_feature_columns": to_drop,
        "keep_features_requested": len(keep_list),
        "keep_features_present": len(keep_present),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "label_mode": args.label_mode,
        "risk_threshold": float(args.risk_threshold),
        "val_fraction": vf,
        "random_state": int(args.random_state),
        "transform_meta": str(meta_path),
        "feature_selection_dir": str(fs_dir),
        "output_columns": list(df_final.columns),
        "output_format": "csv" if args.csv else "parquet",
        "strict_features": bool(args.strict_features),
        "missing_keep_features": missing_keep,
        "match_columns_parquet": str(args.match_columns_parquet.expanduser().resolve())
        if args.match_columns_parquet is not None
        else None,
    }
    (out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"Wrote {train_path} ({len(train_df)} rows), {val_path} ({len(val_df)} rows); "
        f"cols=label+{len(keep_present)} features (no chunk_hash/risk_score)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
