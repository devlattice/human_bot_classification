#!/usr/bin/env python3
"""
LGBM v2 training entrypoint (lower-capacity defaults).

Based on `workspace/model/scripts/lgbm.py`, but exposes model-capacity
and regularization knobs with stricter defaults to reduce overfitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace.model.scripts.lgbm import (  # noqa: E402
    _lgbm_device_kwargs,
    _threshold_sweep,
    evaluate,
    load_parquet_pair,
    plot_roc,
    plot_train_valid_loss,
)


def train_lgbm_b_v2(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    *,
    seed: int,
    device: str,
    log_every: int,
    sample_weight: np.ndarray | None = None,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    min_gain_to_split: float,
    early_stopping_rounds: int,
):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    dev = _lgbm_device_kwargs(device)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        min_child_samples=int(min_child_samples),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        min_gain_to_split=float(min_gain_to_split),
        random_state=int(seed),
        n_jobs=-1,
        verbose=-1,
        **dev,
    )
    callbacks = [early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=True)]
    if log_every > 0:
        callbacks.append(log_evaluation(period=log_every))
    fit_kw: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        callbacks=callbacks,
        **fit_kw,
    )
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train robust lgbm v2 (lower-capacity defaults).")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "_subnet_target" / "dataset" / "robusted_dataset",
        help="Directory with train.parquet and val.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "model" / "artifacts" / "lgbm_b_v2",
        help="Output directory",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip exporting train/valid loss and ROC plots.",
    )
    p.add_argument(
        "--target-human-fpr",
        type=float,
        default=0.05,
        help="Target max human FPR for threshold selection on validation scores.",
    )
    p.add_argument(
        "--threshold-grid-size",
        type=int,
        default=1001,
        help="Number of thresholds for sweep in [0,1].",
    )
    p.add_argument(
        "--threshold-tie-ref",
        type=float,
        default=0.5,
        help="Among max bot_recall under FPR cap, pick threshold closest to this value.",
    )
    p.add_argument(
        "--sample-weight-col",
        default="",
        help="If set, column in train.parquet used as LightGBM sample_weight (val unweighted).",
    )

    # Capacity/regularization defaults stricter than lgbm.py
    p.add_argument("--n-estimators", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--num-leaves", type=int, default=15)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--min-child-samples", type=int, default=400)
    p.add_argument("--subsample", type=float, default=0.6)
    p.add_argument("--colsample-bytree", type=float, default=0.6)
    p.add_argument("--reg-alpha", type=float, default=4.0)
    p.add_argument("--reg-lambda", type=float, default=12.0)
    p.add_argument("--min-gain-to-split", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=150)

    args = p.parse_args()

    sw_col = (args.sample_weight_col or "").strip()
    exclude_feats: tuple[str, ...] = (sw_col,) if sw_col else ()
    train_df, val_df, feature_cols = load_parquet_pair(args.data_dir, exclude_from_features=exclude_feats)
    sample_weight: np.ndarray | None = None
    if sw_col:
        if sw_col not in train_df.columns:
            raise ValueError(f"--sample-weight-col {sw_col!r} not found in train.parquet")
        sample_weight = pd.to_numeric(train_df[sw_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
            raise ValueError(f"Column {sw_col!r} must be finite and nonnegative")
        if float(np.sum(sample_weight)) <= 0:
            raise ValueError(f"Column {sw_col!r} sums to zero")
    X_train = train_df[feature_cols]
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["label"].to_numpy()

    print(
        f"[LGBM_2] data_dir={Path(args.data_dir).resolve()} "
        f"features={len(feature_cols)} rows train={len(train_df)} val={len(val_df)}",
        file=sys.stderr,
        flush=True,
    )

    model = train_lgbm_b_v2(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=args.seed,
        device=args.device,
        log_every=max(0, int(args.log_every)),
        sample_weight=sample_weight,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_gain_to_split=args.min_gain_to_split,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    y_val_score = model.predict_proba(X_val)[:, 1]
    sweep = _threshold_sweep(
        y_true=y_val,
        y_score=y_val_score,
        target_human_fpr=args.target_human_fpr,
        grid_size=args.threshold_grid_size,
        threshold_tie_ref=args.threshold_tie_ref,
    )
    selected_threshold = float(sweep["selected_threshold"])
    val_metrics_at_selected = evaluate(model, X_val, y_val, threshold=selected_threshold)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "model": "LGBM-B-V2",
        "sample_weight_col": sw_col or None,
        "params": {
            "num_leaves": int(args.num_leaves),
            "max_depth": int(args.max_depth),
            "min_child_samples": int(args.min_child_samples),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "min_gain_to_split": float(args.min_gain_to_split),
            "learning_rate": float(args.learning_rate),
            "n_estimators": int(args.n_estimators),
            "early_stopping_rounds": int(args.early_stopping_rounds),
        },
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "threshold_selection": {
            "policy": "max bot_recall under target human_fpr",
            "target_human_fpr": args.target_human_fpr,
            "threshold_tie_ref": float(args.threshold_tie_ref),
            "selected_threshold": selected_threshold,
            "val_metrics_at_selected_threshold": val_metrics_at_selected,
            "sweep": sweep,
        },
        "plots": {"train_valid_loss": [], "roc_curve": []},
    }
    prep_summary_path = Path(args.data_dir).resolve() / "ssl_prepare_summary.json"
    if prep_summary_path.is_file():
        try:
            report["ssl_prepare_summary"] = json.loads(prep_summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            report["ssl_prepare_summary_read_error"] = str(e)

    if not args.no_plots:
        try:
            evals_result = getattr(model, "evals_result_", None) or {}
            report["plots"]["train_valid_loss"] = plot_train_valid_loss(
                evals_result,
                out_dir / "train_valid_loss",
                ["jpg"],
            )
            report["plots"]["roc_curve"] = plot_roc(
                y_true=y_val,
                y_score=y_val_score,
                out_base=out_dir / "roc_curve",
                extensions=["jpg"],
                selected_threshold=selected_threshold,
            )
        except Exception as e:
            print(f"[warn] plot export failed: {e}", file=sys.stderr)
            report["plots"]["error"] = str(e)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols, "n_features": len(feature_cols)}, indent=2),
        encoding="utf-8",
    )

    try:
        import joblib

        joblib.dump(model, out_dir / "lgbm_b_classifier.joblib")
    except Exception as e:
        print(f"[warn] joblib save skipped: {e}", file=sys.stderr)
    model.booster_.save_model(str(out_dir / "lgbm_b_model.txt"))

    print(
        json.dumps(
            {
                "train": train_metrics,
                "val": val_metrics,
                "selected_threshold": selected_threshold,
                "val@selected_threshold": val_metrics_at_selected,
            },
            indent=2,
        )
    )
    print(f"Saved artifacts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
