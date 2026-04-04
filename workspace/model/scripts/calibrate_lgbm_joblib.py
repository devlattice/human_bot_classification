#!/usr/bin/env python3
"""
Post-hoc probability calibration for joblib classifiers (Platt / isotonic).

Designed for model artifacts produced by `workspace/model/scripts/lgbm.py`
and `workspace/model/scripts/lgbm_2.py`:
  - model: lgbm_b_classifier.joblib
  - optional sidecars: feature_cols.json, metrics.json

Goal: keep production decision rule `threshold = 0.5` while making calibrated
probabilities better aligned on validator/holdout-like data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


def _feature_columns_from_artifacts(model_path: Path) -> list[str]:
    out: list[str] = []
    feature_cols_path = model_path.parent / "feature_cols.json"
    if feature_cols_path.is_file():
        payload = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        cols = payload.get("feature_cols")
        if isinstance(cols, list):
            out = [str(c) for c in cols if str(c)]
            if out:
                return out

    metrics_path = model_path.parent / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = payload.get("feature_cols")
        if isinstance(cols, list):
            out = [str(c) for c in cols if str(c)]
    return out


def _feature_columns(model: Any, model_path: Path, df: pd.DataFrame) -> list[str]:
    cols = _feature_columns_from_artifacts(model_path)
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"val parquet missing features from artifacts: {missing[:12]}")
        return cols

    names = getattr(model, "feature_name_in_", None)
    if names is not None and len(names) > 0:
        cols = [str(x) for x in names]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"val parquet missing model features: {missing[:12]}")
        return cols
    if "label" not in df.columns:
        raise ValueError("val parquet must contain `label`")
    return [c for c in df.columns if c != "label"]


def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    human_mask = y == 0
    bot_mask = y == 1
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == y)),
        "human_fpr": float(np.mean(pred[human_mask] == 1)) if np.any(human_mask) else float("nan"),
        "bot_recall": float(np.mean(pred[bot_mask] == 1)) if np.any(bot_mask) else float("nan"),
    }


def _resolve_io_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    artifact_dir = args.artifact_dir.expanduser().resolve() if args.artifact_dir else None
    model_path = args.model_in.expanduser().resolve() if args.model_in else None
    val_path = args.val_parquet.expanduser().resolve() if args.val_parquet else None
    out_path = args.out.expanduser().resolve() if args.out else None

    if artifact_dir is not None:
        model_path = model_path or (artifact_dir / "lgbm_b_classifier.joblib")
        out_path = out_path or (artifact_dir / "lgbm_b_classifier_calibrated.joblib")

    if val_path is None and args.data_dir:
        val_path = args.data_dir.expanduser().resolve() / "val.parquet"

    if model_path is None or val_path is None or out_path is None:
        raise ValueError(
            "Must provide calibration inputs. Either pass --artifact-dir "
            "(optionally with --data-dir), or pass all of "
            "--model-in, --val-parquet, and --out."
        )
    return model_path, val_path, out_path


def main() -> None:
    if joblib is None:
        raise RuntimeError("joblib required: pip install joblib")
    ap = argparse.ArgumentParser(description="Calibrate LGBM joblib on labeled val parquet.")
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Model artifact dir from lgbm/lgbm_2. Defaults: "
        "model=lgbm_b_classifier.joblib, out=lgbm_b_classifier_calibrated.joblib",
    )
    ap.add_argument("--model-in", type=Path, default=None, help="Fitted classifier joblib")
    ap.add_argument("--val-parquet", type=Path, default=None, help="Labeled val parquet")
    ap.add_argument("--out", type=Path, default=None, help="Output calibrated joblib")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="If --val-parquet is omitted, use <data-dir>/val.parquet.",
    )
    ap.add_argument(
        "--method",
        choices=("sigmoid", "isotonic"),
        default="sigmoid",
        help="sigmoid = Platt scaling; isotonic = flexible, needs enough samples per class",
    )
    ap.add_argument(
        "--out-meta",
        type=Path,
        default=None,
        help="Optional JSON with calibration diagnostics (default: <out>.calibration.json).",
    )
    ap.add_argument(
        "--report-threshold",
        type=float,
        default=0.5,
        help="Threshold used in before/after diagnostic block (default: 0.5).",
    )
    args = ap.parse_args()

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, log_loss

    model_path, val_path, out_path = _resolve_io_paths(args)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not val_path.is_file():
        raise FileNotFoundError(val_path)

    base = joblib.load(model_path)
    df = pd.read_parquet(val_path)
    if "label" not in df.columns:
        raise ValueError("val parquet must contain `label` (0/1)")
    feat_cols = _feature_columns(base, model_path, df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    y = df["label"].to_numpy()
    y_float = np.asarray(y, dtype=float)
    if len(np.unique(y[~np.isnan(y_float)])) < 2:
        raise ValueError("val needs both classes for calibration")

    p_raw = base.predict_proba(X)[:, 1]
    brier_before = float(brier_score_loss(y, p_raw))
    ll_before = float(log_loss(y, np.clip(p_raw, 1e-12, 1.0 - 1e-12)))

    # sklearn compatibility:
    # - older versions: CalibratedClassifierCV(..., cv="prefit")
    # - newer versions: prefer FrozenEstimator(base) with cv=None
    try:
        from sklearn.frozen import FrozenEstimator  # type: ignore

        frozen_base = FrozenEstimator(base)
        cal = CalibratedClassifierCV(frozen_base, method=args.method, cv=None)
        cal.fit(X, y)
    except Exception:
        cal = CalibratedClassifierCV(base, method=args.method, cv="prefit")
        cal.fit(X, y)
    p_cal = cal.predict_proba(X)[:, 1]
    brier_after = float(brier_score_loss(y, p_cal))
    ll_after = float(log_loss(y, np.clip(p_cal, 1e-12, 1.0 - 1e-12)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, out_path)

    t = float(args.report_threshold)
    meta = {
        "model_in": str(model_path),
        "val_parquet": str(val_path),
        "feature_source": (
            "feature_cols.json/metrics.json"
            if _feature_columns_from_artifacts(model_path)
            else "model.feature_name_in_ / val columns"
        ),
        "method": args.method,
        "n_val": int(len(df)),
        "n_features": len(feat_cols),
        "class_balance": {
            "n_human_0": int(np.sum(y == 0)),
            "n_bot_1": int(np.sum(y == 1)),
        },
        "brier_score_loss_raw": brier_before,
        "brier_score_loss_calibrated": brier_after,
        "log_loss_raw": ll_before,
        "log_loss_calibrated": ll_after,
        "report_threshold": t,
        "at_report_threshold_raw": _metrics_at_threshold(y, p_raw, t),
        "at_report_threshold_calibrated": _metrics_at_threshold(y, p_cal, t),
        "score_summary_raw": {
            "p01": float(np.quantile(p_raw, 0.01)),
            "p50": float(np.quantile(p_raw, 0.50)),
            "p99": float(np.quantile(p_raw, 0.99)),
            "mean": float(np.mean(p_raw)),
        },
        "score_summary_calibrated": {
            "p01": float(np.quantile(p_cal, 0.01)),
            "p50": float(np.quantile(p_cal, 0.50)),
            "p99": float(np.quantile(p_cal, 0.99)),
            "mean": float(np.mean(p_cal)),
        },
        "out": str(out_path),
    }
    print(json.dumps(meta, indent=2))
    mp = args.out_meta.expanduser().resolve() if args.out_meta else out_path.with_suffix(".calibration.json")
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"Wrote {mp}", file=sys.stderr)


if __name__ == "__main__":
    main()
