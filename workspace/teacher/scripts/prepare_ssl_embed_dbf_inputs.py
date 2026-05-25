#!/usr/bin/env python3
"""
Prepare DBF-augmented inputs for ssl_embed pipeline.

Creates *_with_dbf.parquet variants for common explorer/feature_2 splits and
writes keep_features_with_dbf.txt by appending DBF columns to base keep_features.

Run from repo root:

  PYTHONPATH=. python workspace/teacher/scripts/prepare_ssl_embed_dbf_inputs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TEACHER_ROOT = Path(__file__).resolve().parents[1]
if str(TEACHER_ROOT) not in sys.path:
    sys.path.insert(0, str(TEACHER_ROOT))

from dbf.drift import assess_dbf_column_stability, save_dbf_drift_report  # noqa: E402
from dbf.features import (  # noqa: E402
    DBF_COLUMN_NAMES,
    add_dbf_columns,
    fit_dbf_quantile_bounds,
    load_dbf_quantile_bounds,
    normalize_dbf_column_subset,
    save_dbf_quantile_bounds,
)


def _default_paths() -> dict[str, Path]:
    base = (
        REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "data"
    )
    return {
        "train": base / "public" / "train.parquet",
        "val": base / "public" / "val.parquet",
        "validator": base / "validator" / "validator.parquet",
        "pub1": base / "test" / "pb_1.parquet",
        "pub2": base / "test" / "pb_2.parquet",
        "holdout1": base / "test" / "holdout_1.parquet",
        "holdout2": base / "test" / "holdout_2.parquet",
        "irc_val": base / "irc" / "irc_val.parquet",
    }


def _with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _write_dbf_parquet(
    inp: Path,
    out: Path,
    overwrite_existing_dbf: bool,
    quantile_bounds: dict[str, tuple[float, float]] | None,
    dbf_columns: tuple[str, ...],
) -> tuple[int, int]:
    df = pd.read_parquet(inp)
    existing = [c for c in DBF_COLUMN_NAMES if c in df.columns]
    if existing and not overwrite_existing_dbf:
        raise SystemExit(
            f"{inp} already contains DBF columns {existing}; re-run with --overwrite-existing-dbf."
        )
    if existing:
        df = df.drop(columns=list(DBF_COLUMN_NAMES), errors="ignore")
    df = add_dbf_columns(
        df, inplace=False, quantile_bounds=quantile_bounds, dbf_columns=dbf_columns
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df), len(df.columns)


def _write_features_with_dbf(
    base_features: Path, out_features: Path, dbf_columns: tuple[str, ...]
) -> None:
    lines = [ln.strip() for ln in base_features.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for c in dbf_columns:
        if c not in lines:
            lines.append(c)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_features.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    defaults = _default_paths()
    p = argparse.ArgumentParser(description="Create *_with_dbf.parquet inputs for ssl_embed.")
    p.add_argument(
        "--base-features-txt",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "config"
        / "keep_features.txt",
    )
    p.add_argument(
        "--out-features-txt",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "config"
        / "keep_features_with_dbf.txt",
    )
    p.add_argument("--suffix", type=str, default="_with_dbf")
    p.add_argument("--overwrite-existing-dbf", action="store_true")
    p.add_argument("--strict", action="store_true", help="Fail if any default input parquet is missing.")
    p.add_argument(
        "--quantile-bounds-json",
        type=Path,
        default=None,
        help="Read/write train-fitted DBF winsor bounds (default: feature_2/config/dbf_quantile_bounds.json).",
    )
    p.add_argument(
        "--dbf-drift-prune",
        action="store_true",
        help="Drop DBF columns with large train→val/test mean shift (see --dbf-drift-*).",
    )
    p.add_argument(
        "--dbf-drift-max-mean-z",
        type=float,
        default=3.0,
        help="Max |mean split − mean train| / std(train) per DBF column (default: 3).",
    )
    p.add_argument(
        "--dbf-drift-min-train-std",
        type=float,
        default=1e-5,
        help="Mark unstable if train std after DBF transform is below this (default: 1e-5).",
    )
    p.add_argument(
        "--dbf-drift-report-json",
        type=Path,
        default=None,
        help="Write drift JSON next to quantile bounds by default.",
    )
    args = p.parse_args()

    default_bounds = (
        REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "config"
        / "dbf_quantile_bounds.json"
    )
    bounds_path = (args.quantile_bounds_json or default_bounds).expanduser().resolve()

    train_inp = defaults["train"].expanduser().resolve()
    val_inp = defaults["val"].expanduser().resolve()
    quantile_bounds: dict[str, tuple[float, float]] | None = None
    if train_inp.is_file():
        df_train = pd.read_parquet(train_inp)
        quantile_bounds = fit_dbf_quantile_bounds(df_train)
        save_dbf_quantile_bounds(bounds_path, quantile_bounds)
        print(f"[dbf] fitted quantile bounds from train -> {bounds_path}")
    elif bounds_path.is_file():
        quantile_bounds = load_dbf_quantile_bounds(bounds_path)
        print(f"[dbf] loaded quantile bounds from {bounds_path}")
    else:
        print(
            "[dbf] warning: no train parquet and no bounds JSON; DBF uses per-frame quantiles (split leakage)."
        )

    dbf_columns = normalize_dbf_column_subset(None)
    drift_report_path = (
        args.dbf_drift_report_json.expanduser().resolve()
        if args.dbf_drift_report_json is not None
        else bounds_path.parent / "dbf_drift_report.json"
    )
    if args.dbf_drift_prune:
        if quantile_bounds is None:
            raise SystemExit("[dbf] --dbf-drift-prune requires train-fitted or loaded quantile bounds.")
        if not train_inp.is_file():
            raise SystemExit(f"[dbf] --dbf-drift-prune requires train parquet: {train_inp}")
        if not val_inp.is_file():
            raise SystemExit(f"[dbf] --dbf-drift-prune requires val parquet: {val_inp}")
        df_train_drift = pd.read_parquet(train_inp)
        df_val = pd.read_parquet(val_inp)
        test_frames: dict[str, pd.DataFrame] = {}
        for key in ("pub1", "pub2", "holdout1", "holdout2"):
            pth = defaults[key].expanduser().resolve()
            if pth.is_file():
                test_frames[key] = pd.read_parquet(pth)
        stable, unstable, detail = assess_dbf_column_stability(
            train_df=df_train_drift,
            val_df=df_val,
            test_frames=test_frames,
            quantile_bounds=quantile_bounds,
            max_abs_mean_z=float(args.dbf_drift_max_mean_z),
            min_train_std=float(args.dbf_drift_min_train_std),
        )
        save_dbf_drift_report(drift_report_path, detail)
        print(f"[dbf] drift report -> {drift_report_path}")
        if unstable:
            print(f"[dbf] drift prune removing columns: {list(unstable)}")
        else:
            print("[dbf] drift prune: all DBF columns stable")
        if not stable:
            raise SystemExit("[dbf] drift prune removed all DBF columns; relax --dbf-drift-max-mean-z or checks.")
        quantile_bounds = {k: quantile_bounds[k] for k in stable}
        save_dbf_quantile_bounds(bounds_path, quantile_bounds)
        print(f"[dbf] updated quantile bounds ({len(stable)} cols) -> {bounds_path}")
        dbf_columns = stable

    created: list[tuple[str, Path]] = []
    skipped: list[tuple[str, Path]] = []
    for name, inp in defaults.items():
        inp = inp.expanduser().resolve()
        if not inp.is_file():
            if args.strict:
                raise SystemExit(f"Missing required input parquet: {inp}")
            skipped.append((name, inp))
            continue
        out = _with_suffix(inp, args.suffix)
        rows, cols = _write_dbf_parquet(
            inp, out, args.overwrite_existing_dbf, quantile_bounds, dbf_columns
        )
        print(f"[dbf] {name}: {inp} -> {out} rows={rows} cols={cols}")
        created.append((name, out))

    base_features = args.base_features_txt.expanduser().resolve()
    if not base_features.is_file():
        raise SystemExit(f"Missing --base-features-txt: {base_features}")
    out_features = args.out_features_txt.expanduser().resolve()
    _write_features_with_dbf(base_features, out_features, dbf_columns)
    print(f"[dbf] features: {base_features} -> {out_features} (+{len(dbf_columns)} dbf columns)")

    if skipped:
        print("[dbf] skipped missing inputs:")
        for name, path in skipped:
            print(f"  - {name}: {path}")
    print(f"[dbf] created {len(created)} parquet files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

