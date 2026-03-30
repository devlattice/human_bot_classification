#!/usr/bin/env python3
"""
LGBM (regularized) training entrypoint.

Uses the same parquet schema as LGBM.py:
  - train.parquet and val.parquet with `label` + feature columns
  - all non-label columns are used as features

Default data dir points to lgbm_train_sharded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace.model.LGBM import (  # noqa: E402
    evaluate,
    load_parquet_pair,
    plot_roc,
    plot_train_valid_loss,
)


def _threshold_sweep(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_human_fpr: float,
    grid_size: int = 1001,
) -> Dict[str, Any]:
    """
    Sweep thresholds and pick the best one under a human-FPR constraint.

    Selection rule:
      1) human_fpr <= target_human_fpr
      2) maximize bot_recall
      3) tie-break: higher threshold (more conservative on humans)
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    human_mask = y == 0
    bot_mask = y == 1

    thresholds = np.linspace(0.0, 1.0, grid_size)
    rows = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        human_fpr = (
            float(np.mean(pred[human_mask] == 1)) if np.any(human_mask) else float("nan")
        )
        bot_recall = (
            float(np.mean(pred[bot_mask] == 1)) if np.any(bot_mask) else float("nan")
        )
        acc = float(np.mean(pred == y)) if len(y) else float("nan")
        rows.append(
            {
                "threshold": float(t),
                "human_fpr": human_fpr,
                "bot_recall": bot_recall,
                "accuracy": acc,
            }
        )

    feasible = [
        r for r in rows if np.isfinite(r["human_fpr"]) and r["human_fpr"] <= target_human_fpr
    ]
    if feasible:
        best = max(feasible, key=lambda r: (r["bot_recall"], r["threshold"]))
        hit_target = True
    else:
        min_fpr = min(r["human_fpr"] for r in rows if np.isfinite(r["human_fpr"]))
        tied = [r for r in rows if r["human_fpr"] == min_fpr]
        best = max(tied, key=lambda r: (r["bot_recall"], r["threshold"]))
        hit_target = False

    checkpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    summary_points: Dict[str, Dict[str, float]] = {}
    for cp in checkpoints:
        idx = int(round(cp * (grid_size - 1)))
        rp = rows[idx]
        summary_points[f"{cp:.1f}"] = {
            "human_fpr": rp["human_fpr"],
            "bot_recall": rp["bot_recall"],
            "accuracy": rp["accuracy"],
        }

    return {
        "target_human_fpr": float(target_human_fpr),
        "grid_size": int(grid_size),
        "hit_target": bool(hit_target),
        "selected_threshold": best["threshold"],
        "selected_metrics": {
            "human_fpr": best["human_fpr"],
            "bot_recall": best["bot_recall"],
            "accuracy": best["accuracy"],
        },
        "checkpoints": summary_points,
    }


def _lgbm_device_kwargs(device: str) -> Dict[str, Any]:
    d = (device or "cpu").strip().lower()
    if d == "gpu":
        return {"device": "gpu", "gpu_device_id": 0, "gpu_platform_id": 0}
    return {"device": "cpu"}


def train_lgbm_b(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    *,
    seed: int,
    device: str,
    log_every: int,
):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    dev = _lgbm_device_kwargs(device)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=200,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **dev,
    )
    callbacks = [early_stopping(stopping_rounds=120, verbose=True)]
    if log_every > 0:
        callbacks.append(log_evaluation(period=log_every))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train robust lgbm on chunk parquet.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "_subnet_target" / "dataset" / "robusted_dataset",
        help="Directory with train.parquet and val.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "model" / "artifacts" / "lgbm_b",
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
    args = p.parse_args()

    train_df, val_df, feature_cols = load_parquet_pair(args.data_dir)
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

    model = train_lgbm_b(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=args.seed,
        device=args.device,
        log_every=max(0, int(args.log_every)),
    )

    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    y_val_score = model.predict_proba(X_val)[:, 1]
    sweep = _threshold_sweep(
        y_true=y_val,
        y_score=y_val_score,
        target_human_fpr=args.target_human_fpr,
        grid_size=args.threshold_grid_size,
    )
    selected_threshold = float(sweep["selected_threshold"])
    val_metrics_at_selected = evaluate(model, X_val, y_val, threshold=selected_threshold)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "model": "LGBM-B",
        "params": {
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 200,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "learning_rate": 0.03,
            "n_estimators": 3000,
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
            "selected_threshold": selected_threshold,
            "val_metrics_at_selected_threshold": val_metrics_at_selected,
            "sweep": sweep,
        },
        "plots": {"train_valid_loss": [], "roc_curve": []},
    }
    if not args.no_plots:
        try:
            evals_result = getattr(model, "evals_result_", None) or {}
            report["plots"]["train_valid_loss"] = plot_train_valid_loss(
                evals_result,
                out_dir / "train_valid_loss",
                ["jpg"],  # requested: JPG only
            )
            report["plots"]["roc_curve"] = plot_roc(
                y_true=y_val,
                y_score=y_val_score,
                out_base=out_dir / "roc_curve",
                extensions=["jpg"],  # requested: JPG only
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
