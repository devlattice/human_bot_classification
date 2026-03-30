#!/usr/bin/env python3
"""
Robust feature preprocessing for parquet train/val datasets.

Fit on train only, apply to train and val:
  - quantile clipping (winsorization)
  - optional log1p on selected nonnegative heavy-tail features

Use this to reduce score saturation and domain-shift sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

_FEATURES_DIR = Path(__file__).resolve().parent / "features"


def _resolve_split_parquet_paths(data_dir: Path, split: str) -> List[Path]:
    split_path = data_dir / f"{split}.parquet"
    if split_path.is_file():
        return [split_path]

    split_dir = data_dir / split
    if split_dir.is_dir():
        nested = sorted(split_dir.glob(f"{split}*.parquet"))
        if nested:
            return nested

    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        js = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = "train_paths" if split == "train" else "val_paths"
        raw = js.get(key, [])
        if isinstance(raw, list):
            out: List[Path] = []
            for item in raw:
                p = Path(str(item))
                if not p.is_absolute():
                    p = p if p.exists() else (data_dir / p)
                if p.is_file() and p.suffix.lower() == ".parquet":
                    out.append(p.resolve())
            if out:
                return out
    return []


def _load_split_df(data_dir: Path, split: str) -> pd.DataFrame:
    paths = _resolve_split_parquet_paths(data_dir, split)
    if not paths:
        raise FileNotFoundError(
            f"{data_dir}: could not resolve {split} parquet files "
            f"(expected {split}.parquet, {split}/, or manifest {split}_paths)."
        )
    if len(paths) == 1:
        return pd.read_parquet(paths[0])
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _load_pair(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _load_split_df(data_dir, "train")
    val_df = _load_split_df(data_dir, "val")
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError("Both train and val must contain `label`")
    return train_df, val_df


def _coerce_numeric_pair(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_train = train_df.copy()
    out_val = val_df.copy()
    for c in feature_cols:
        out_train[c] = pd.to_numeric(out_train[c], errors="coerce").astype(float)
        out_val[c] = pd.to_numeric(out_val[c], errors="coerce").astype(float)
    return out_train, out_val


def _default_log1p_candidates(columns: List[str]) -> List[str]:
    keys = ("pot", "stack", "norm_bb")
    out: List[str] = []
    for c in columns:
        cl = c.lower()
        if any(k in cl for k in keys):
            out.append(c)
    return sorted(out)


def _read_feature_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.append(s)
    return items


def _maybe_load_default_feature_list(filename: str, feature_cols: List[str]) -> List[str]:
    p = _FEATURES_DIR / filename
    if not p.is_file():
        return []
    return _existing_features(_read_feature_list(p), feature_cols)


def _existing_features(items: List[str], feature_cols: List[str]) -> List[str]:
    pool = set(feature_cols)
    return sorted({x for x in items if x in pool})


def _fit_quantiles(train_df: pd.DataFrame, feature_cols: List[str], q_low: float, q_high: float) -> Dict[str, Dict[str, float]]:
    bounds: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        s = pd.to_numeric(train_df[c], errors="coerce").astype(float)
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        lo = float(np.quantile(finite, q_low))
        hi = float(np.quantile(finite, q_high))
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = {"low": lo, "high": hi}
    return bounds


def _apply_clipping(df: pd.DataFrame, bounds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, b in bounds.items():
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        out[c] = s.clip(lower=b["low"], upper=b["high"])
    return out


def _fit_log1p_allowlist(
    train_df: pd.DataFrame, candidate_cols: List[str], strict_nonnegative: bool
) -> List[str]:
    selected: List[str] = []
    for c in candidate_cols:
        if c not in train_df.columns:
            continue
        s = pd.to_numeric(train_df[c], errors="coerce").astype(float)
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        if strict_nonnegative:
            if float(finite.min()) >= 0.0:
                selected.append(c)
        else:
            # allow tiny negatives from numeric noise
            if float(finite.min()) >= -1e-9:
                selected.append(c)
    return sorted(set(selected))


def _apply_log1p(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        s = np.maximum(s, 0.0)
        out[c] = np.log1p(s)
    return out


def _fit_robust_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        s = pd.to_numeric(train_df[c], errors="coerce").astype(float)
        finite = s[np.isfinite(s)]
        if finite.empty:
            continue
        q1 = float(np.quantile(finite, 0.25))
        q3 = float(np.quantile(finite, 0.75))
        iqr = q3 - q1
        if iqr == 0.0:
            iqr = 1.0
        stats[c] = {"median": float(np.median(finite)), "iqr": float(iqr)}
    return stats


def _apply_robust_scaler(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, st in stats.items():
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        out[c] = (s - st["median"]) / st["iqr"]
    return out


def _apply_abs_clip(df: pd.DataFrame, feature_cols: List[str], abs_value: float) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        out[c] = s.clip(lower=-abs_value, upper=abs_value)
    return out


def _drop_rows_with_too_many_nans(
    df: pd.DataFrame, feature_cols: List[str], frac_over: float
) -> Tuple[pd.DataFrame, int]:
    if not (0.0 <= frac_over <= 1.0):
        raise ValueError("--drop-row-nan-frac-over must be in [0, 1]")
    if not feature_cols:
        return df, 0
    frac = df[feature_cols].isna().mean(axis=1)
    keep = frac <= frac_over
    dropped = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), dropped


def _fill_remaining_nans(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    med = train_df[feature_cols].median(numeric_only=True).to_dict()
    med = {str(k): float(v) for k, v in med.items() if np.isfinite(v)}
    train_out = train_df.copy()
    val_out = val_df.copy()
    train_out[feature_cols] = train_out[feature_cols].fillna(med).fillna(0.0)
    val_out[feature_cols] = val_out[feature_cols].fillna(med).fillna(0.0)
    return train_out, val_out, med


def _compute_fill_medians(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    med = df[feature_cols].median(numeric_only=True).to_dict()
    return {str(k): float(v) for k, v in med.items() if np.isfinite(v)}


def _fill_nans_with_medians(df: pd.DataFrame, feature_cols: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    fill_map = {k: float(v) for k, v in medians.items() if k in feature_cols and np.isfinite(v)}
    out[feature_cols] = out[feature_cols].fillna(fill_map).fillna(0.0)
    return out


def _resolve_feature_cols_from_meta(meta: Dict[str, object], df_cols: List[str]) -> List[str]:
    if "clip_bounds" in meta and isinstance(meta["clip_bounds"], dict):
        cols = [str(c) for c in cast(dict, meta["clip_bounds"]).keys()]
        if cols:
            return [c for c in cols if c in df_cols]
    if "fillna" in meta and isinstance(meta["fillna"], dict):
        fillna = cast(dict, meta["fillna"])
        medians = fillna.get("medians")
        if isinstance(medians, dict):
            cols = [str(c) for c in medians.keys()]
            if cols:
                return [c for c in cols if c in df_cols]
    return [c for c in df_cols if c != "label"]


def _apply_from_meta(df: pd.DataFrame, meta: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    df_cols = list(out.columns)
    feature_cols = _resolve_feature_cols_from_meta(meta, df_cols)
    if "label" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "label"]
    out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    clip_meta = meta.get("clip", {})
    clip_enabled = isinstance(clip_meta, dict) and bool(cast(dict, clip_meta).get("enabled", False))
    clip_bounds = meta.get("clip_bounds", {})
    if clip_enabled and isinstance(clip_bounds, dict) and clip_bounds:
        out = _apply_clipping(out, cast(Dict[str, Dict[str, float]], clip_bounds))

    log_meta = meta.get("log1p", {})
    log_enabled = isinstance(log_meta, dict) and bool(cast(dict, log_meta).get("enabled", False))
    log_cols = meta.get("log1p_selected_features", [])
    if log_enabled and isinstance(log_cols, list) and log_cols:
        out = _apply_log1p(out, [str(c) for c in log_cols])

    robust_meta = meta.get("robust_scale", {})
    robust_enabled = isinstance(robust_meta, dict) and bool(cast(dict, robust_meta).get("enabled", False))
    robust_stats = meta.get("robust_scale_stats", {})
    if robust_enabled and isinstance(robust_stats, dict) and robust_stats:
        out = _apply_robust_scaler(out, cast(Dict[str, Dict[str, float]], robust_stats))
        scaled_clip_abs = float(cast(dict, robust_meta).get("scaled_clip_abs", 0.0))
        if scaled_clip_abs > 0.0:
            out = _apply_abs_clip(out, list(cast(dict, robust_stats).keys()), scaled_clip_abs)

    fillna_meta = meta.get("fillna", {})
    medians: Dict[str, float] = {}
    if isinstance(fillna_meta, dict):
        raw_medians = cast(dict, fillna_meta).get("medians", {})
        if isinstance(raw_medians, dict):
            medians = {str(k): float(v) for k, v in raw_medians.items() if np.isfinite(v)}
    out = _fill_nans_with_medians(out, feature_cols, medians)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantile clip + optional log1p transforms for train/val parquet.")
    ap.add_argument("--data-dir", help="Input dataset dir with train.parquet + val.parquet")
    ap.add_argument("--out-dir", help="Output directory for transformed train/val")
    ap.add_argument("--q-low", type=float, default=0.01, help="Lower quantile for clipping (default 0.01)")
    ap.add_argument("--q-high", type=float, default=0.99, help="Upper quantile for clipping (default 0.99)")
    ap.add_argument("--disable-clipping", action="store_true", help="Skip quantile clipping")
    ap.add_argument("--enable-log1p", action="store_true", help="Apply log1p transform to selected features")
    ap.add_argument(
        "--log1p-all-nonnegative",
        action="store_true",
        help="Experimental: apply log1p to all features whose train values are nonnegative",
    )
    ap.add_argument(
        "--log1p-features-file",
        help="Optional text file: one feature per line for log1p candidates. Default uses pot/stack/norm_bb name match.",
    )
    ap.add_argument("--log1p-feature", action="append", default=[], help="Additional log1p candidate feature (repeatable)")
    ap.add_argument("--strict-nonnegative-log1p", action="store_true", help="Only allow log1p if train min >= 0")
    ap.add_argument(
        "--keep-features-file",
        help="Optional file generated from stats (one feature per line). Used for metadata or with --restrict-to-keep-features.",
    )
    ap.add_argument(
        "--heavy-transform-features-file",
        help="Optional file of high-shift features. If set, log1p/robust-scale target these features by default.",
    )
    ap.add_argument(
        "--regularize-features-file",
        help="Optional file of features requiring stronger regularization; merged with heavy-transform list for scaling.",
    )
    ap.add_argument(
        "--restrict-to-keep-features",
        action="store_true",
        help="If set with --keep-features-file, output only keep-features + label (drops others).",
    )
    ap.add_argument("--enable-robust-scale", action="store_true", help="Apply (x - median) / IQR per feature")
    ap.add_argument(
        "--scaled-clip-abs",
        type=float,
        default=0.0,
        help="If >0 and robust-scale enabled, clip scaled features to [-abs, +abs]",
    )
    ap.add_argument(
        "--drop-row-nan-frac-over",
        type=float,
        default=0.20,
        help="Drop rows with NaN fraction over this threshold (features only). Set <0 to disable.",
    )
    ap.add_argument(
        "--fit-stats-from",
        help="Optional parquet path used only to fit transform stats "
        "(clip/log1p/robust-scale/fill medians).",
    )
    ap.add_argument(
        "--transform-meta-in",
        help="Apply-only mode: path to an existing transform_meta.json generated from train fit.",
    )
    ap.add_argument(
        "--in-parquet",
        help="Apply-only mode: input parquet file to transform using --transform-meta-in.",
    )
    ap.add_argument(
        "--out-parquet",
        help="Apply-only mode: output parquet path for transformed --in-parquet.",
    )
    args = ap.parse_args()

    apply_mode = bool(args.transform_meta_in or args.in_parquet or args.out_parquet)
    if apply_mode:
        if not (args.transform_meta_in and args.in_parquet and args.out_parquet):
            raise ValueError("Apply-only mode requires --transform-meta-in, --in-parquet, and --out-parquet")
        meta_path = Path(args.transform_meta_in).expanduser().resolve()
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        src_path = Path(args.in_parquet).expanduser().resolve()
        if not src_path.is_file():
            raise FileNotFoundError(src_path)
        dst_path = Path(args.out_parquet).expanduser().resolve()
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        df = pd.read_parquet(src_path)
        out_df = _apply_from_meta(df, meta)
        out_df.to_parquet(dst_path, index=False)
        print(f"[robust_feature_transform] apply-only input={src_path}")
        print(f"[robust_feature_transform] apply-only meta={meta_path}")
        print(f"[robust_feature_transform] apply-only output={dst_path}")
        print(f"[robust_feature_transform] rows={len(df)} cols={len(df.columns)}")
        return

    if not args.data_dir or not args.out_dir:
        raise ValueError("Fit mode requires both --data-dir and --out-dir")

    if not (0.0 <= args.q_low < args.q_high <= 1.0):
        raise ValueError("Require 0 <= q-low < q-high <= 1")

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = _load_pair(data_dir)
    feature_cols = [c for c in train_df.columns if c != "label"]
    if set(feature_cols) != set([c for c in val_df.columns if c != "label"]):
        raise ValueError("Train/val feature columns differ")
    train_df, val_df = _coerce_numeric_pair(train_df, val_df, feature_cols)
    fit_df = train_df
    if args.fit_stats_from:
        fit_path = Path(args.fit_stats_from).expanduser().resolve()
        if not fit_path.is_file():
            raise FileNotFoundError(fit_path)
        fit_df = pd.read_parquet(fit_path)
        if "label" not in fit_df.columns:
            raise ValueError("--fit-stats-from parquet must contain `label`")
        missing = [c for c in feature_cols if c not in fit_df.columns]
        if missing:
            raise ValueError(
                "--fit-stats-from is missing required feature columns, "
                f"first missing: {missing[:8]}"
            )
        fit_df = fit_df[["label"] + feature_cols].copy()
        fit_df[feature_cols] = fit_df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    keep_features: List[str] = []
    heavy_features: List[str] = []
    regularize_features: List[str] = []
    if args.keep_features_file:
        keep_features = _existing_features(
            _read_feature_list(Path(args.keep_features_file).expanduser().resolve()),
            feature_cols,
        )
    else:
        keep_features = _maybe_load_default_feature_list("keep_features.txt", feature_cols)
    if args.heavy_transform_features_file:
        heavy_features = _existing_features(
            _read_feature_list(Path(args.heavy_transform_features_file).expanduser().resolve()),
            feature_cols,
        )
    else:
        heavy_features = _maybe_load_default_feature_list("heavy_transform_features.txt", feature_cols)
    if args.regularize_features_file:
        regularize_features = _existing_features(
            _read_feature_list(Path(args.regularize_features_file).expanduser().resolve()),
            feature_cols,
        )
    else:
        regularize_features = _maybe_load_default_feature_list("regularize_features.txt", feature_cols)

    train_rows_before = int(len(train_df))
    val_rows_before = int(len(val_df))
    drop_thresh = float(args.drop_row_nan_frac_over)
    dropped_train = 0
    dropped_val = 0
    if drop_thresh >= 0.0:
        train_df, dropped_train = _drop_rows_with_too_many_nans(train_df, feature_cols, drop_thresh)
        val_df, dropped_val = _drop_rows_with_too_many_nans(val_df, feature_cols, drop_thresh)

    transform_meta: Dict[str, object] = {
        "input_data_dir": str(data_dir),
        "output_data_dir": str(out_dir),
        "fit_stats_from": str(Path(args.fit_stats_from).expanduser().resolve()) if args.fit_stats_from else None,
        "n_features": int(len(feature_cols)),
        "clip": {"enabled": not args.disable_clipping, "q_low": args.q_low, "q_high": args.q_high},
        "log1p": {"enabled": bool(args.enable_log1p)},
        "robust_scale": {"enabled": bool(args.enable_robust_scale), "scaled_clip_abs": float(args.scaled_clip_abs)},
        "feature_lists": {
            "keep_count": int(len(keep_features)),
            "heavy_transform_count": int(len(heavy_features)),
            "regularize_count": int(len(regularize_features)),
        },
        "row_nan_drop_threshold": None if drop_thresh < 0.0 else float(drop_thresh),
        "rows": {
            "train_before": train_rows_before,
            "val_before": val_rows_before,
            "train_dropped_too_many_nans": dropped_train,
            "val_dropped_too_many_nans": dropped_val,
            "train_after_drop": int(len(train_df)),
            "val_after_drop": int(len(val_df)),
        },
    }

    if not args.disable_clipping:
        clip_bounds = _fit_quantiles(fit_df, feature_cols, args.q_low, args.q_high)
        train_df = _apply_clipping(train_df, clip_bounds)
        val_df = _apply_clipping(val_df, clip_bounds)
        transform_meta["clip_bounds"] = clip_bounds
        transform_meta["clip_features_count"] = int(len(clip_bounds))
    else:
        transform_meta["clip_bounds"] = {}
        transform_meta["clip_features_count"] = 0

    log1p_selected: List[str] = []
    if args.log1p_all_nonnegative and not args.enable_log1p:
        raise ValueError("--log1p-all-nonnegative requires --enable-log1p")

    if args.enable_log1p:
        candidates: List[str]
        if args.log1p_all_nonnegative:
            candidates = list(feature_cols)
        elif heavy_features:
            candidates = list(heavy_features)
        elif args.log1p_features_file:
            candidates = _read_feature_list(Path(args.log1p_features_file).expanduser().resolve())
        else:
            candidates = _default_log1p_candidates(feature_cols)
        if args.log1p_feature:
            candidates.extend(args.log1p_feature)
        candidates = sorted(set(candidates))

        log1p_selected = _fit_log1p_allowlist(
            fit_df,
            candidates,
            strict_nonnegative=bool(args.strict_nonnegative_log1p),
        )
        train_df = _apply_log1p(train_df, log1p_selected)
        val_df = _apply_log1p(val_df, log1p_selected)

    transform_meta["log1p_all_nonnegative"] = bool(args.log1p_all_nonnegative)
    transform_meta["log1p_selected_features"] = log1p_selected
    transform_meta["log1p_selected_count"] = int(len(log1p_selected))

    if args.enable_robust_scale:
        scale_targets = sorted(set(heavy_features + regularize_features)) if (heavy_features or regularize_features) else feature_cols
        robust_stats = _fit_robust_scaler(fit_df, scale_targets)
        train_df = _apply_robust_scaler(train_df, robust_stats)
        val_df = _apply_robust_scaler(val_df, robust_stats)
        transform_meta["robust_scale_stats"] = robust_stats
        transform_meta["robust_scale_features_count"] = int(len(robust_stats))
        if args.scaled_clip_abs > 0.0:
            train_df = _apply_abs_clip(train_df, list(robust_stats.keys()), float(args.scaled_clip_abs))
            val_df = _apply_abs_clip(val_df, list(robust_stats.keys()), float(args.scaled_clip_abs))
    else:
        transform_meta["robust_scale_stats"] = {}
        transform_meta["robust_scale_features_count"] = 0

    fill_medians = _compute_fill_medians(fit_df, feature_cols)
    train_df = _fill_nans_with_medians(train_df, feature_cols, fill_medians)
    val_df = _fill_nans_with_medians(val_df, feature_cols, fill_medians)
    if args.restrict_to_keep_features:
        if not keep_features:
            raise ValueError("--restrict-to-keep-features requires --keep-features-file with at least one valid feature")
        keep_cols = ['label'] + keep_features
        train_df = train_df[keep_cols]
        val_df = val_df[keep_cols]
        transform_meta["restricted_to_keep_features"] = True
        transform_meta["restricted_feature_count"] = int(len(keep_features))
    else:
        transform_meta["restricted_to_keep_features"] = False

    transform_meta["fillna"] = {
        "method": "train_median_then_zero",
        "median_features_count": int(len(fill_medians)),
        "medians": fill_medians,
    }

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    (out_dir / "transform_meta.json").write_text(json.dumps(transform_meta, indent=2), encoding="utf-8")

    print(f"[robust_feature_transform] input={data_dir}")
    print(f"[robust_feature_transform] output={out_dir}")
    print(
        "[robust_feature_transform] clipping="
        f"{'on' if not args.disable_clipping else 'off'} "
        f"q=({args.q_low},{args.q_high}) features={transform_meta['clip_features_count']}"
    )
    print(
        "[robust_feature_transform] log1p="
        f"{'on' if args.enable_log1p else 'off'} "
        f"features={transform_meta['log1p_selected_count']}"
    )
    print(
        "[robust_feature_transform] robust_scale="
        f"{'on' if args.enable_robust_scale else 'off'} "
        f"features={transform_meta['robust_scale_features_count']} "
        f"scaled_clip_abs={args.scaled_clip_abs}"
    )
    print(
        "[robust_feature_transform] rows "
        f"train {train_rows_before}->{len(train_df)} "
        f"val {val_rows_before}->{len(val_df)} "
        f"(drop_nan_frac_over={drop_thresh})"
    )
    print(f"[robust_feature_transform] wrote: {out_dir / 'transform_meta.json'}")


if __name__ == "__main__":
    main()
