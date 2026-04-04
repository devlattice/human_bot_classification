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
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_parquet_pair(
    data_dir: Path,
    *,
    exclude_from_features: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data_dir = Path(data_dir).expanduser().resolve()
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"{data_dir}: expected train.parquet and val.parquet")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError("Both train and val must contain `label` column")
    skip = {"label", *exclude_from_features}
    feature_cols = [c for c in train_df.columns if c not in skip]
    val_feats = [c for c in val_df.columns if c not in skip]
    if set(feature_cols) != set(val_feats):
        raise ValueError("Train/val feature columns differ (after excluding label and weight columns)")
    return train_df, val_df, feature_cols


def evaluate(model, X, y_true, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    y_pred = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "human_fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "bot_tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
    return out


def plot_train_valid_loss(evals_result: Dict[str, Any], out_base: Path, extensions: list[str]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    train_vals = (((evals_result or {}).get("train") or {}).get("binary_logloss")) or []
    valid_vals = (((evals_result or {}).get("valid") or {}).get("binary_logloss")) or []
    if not train_vals and not valid_vals:
        return []
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_vals:
        ax.plot(np.arange(1, len(train_vals) + 1), train_vals, label="train")
    if valid_vals:
        ax.plot(np.arange(1, len(valid_vals) + 1), valid_vals, label="valid")
    ax.set_title("Train/Valid Binary Logloss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Logloss")
    ax.legend()
    fig.tight_layout()
    out_files: list[str] = []
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in extensions:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=170)
        out_files.append(str(p))
    plt.close(fig)
    return out_files


def plot_roc(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_base: Path,
    extensions: list[str],
    selected_threshold: float | None = None,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return []
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(fpr, tpr, label=f"ROC AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    if selected_threshold is not None:
        pred = (y_score >= float(selected_threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        th_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        th_tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        if np.isfinite(th_fpr) and np.isfinite(th_tpr):
            ax.scatter([th_fpr], [th_tpr], color="red", s=36, label=f"threshold={selected_threshold:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_files: list[str] = []
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in extensions:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=170)
        out_files.append(str(p))
    plt.close(fig)
    return out_files


def _threshold_sweep(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_human_fpr: float,
    grid_size: int = 1001,
    threshold_tie_ref: float = 0.5,
) -> Dict[str, Any]:
    """
    Sweep thresholds and pick the best one under a human-FPR constraint.

    Selection rule:
      1) human_fpr <= target_human_fpr
      2) maximize bot_recall
      3) tie-break: threshold closest to ``threshold_tie_ref`` (default 0.5); then
         lower threshold if still tied (stable grid behavior vs always ~0 or ~1)
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
    ref = float(threshold_tie_ref)

    def _pick_closest(recall_pool: list[Dict[str, Any]]) -> Dict[str, Any]:
        finite = [r for r in recall_pool if np.isfinite(r["bot_recall"])]
        if not finite:
            return min(recall_pool, key=lambda r: (abs(r["threshold"] - ref), r["threshold"]))
        max_recall = max(r["bot_recall"] for r in finite)
        winners = [r for r in finite if r["bot_recall"] == max_recall]
        return min(winners, key=lambda r: (abs(r["threshold"] - ref), r["threshold"]))

    if feasible:
        best = _pick_closest(feasible)
        hit_target = True
    else:
        min_fpr = min(r["human_fpr"] for r in rows if np.isfinite(r["human_fpr"]))
        tied = [r for r in rows if r["human_fpr"] == min_fpr]
        best = _pick_closest(tied)
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
        "threshold_tie_ref": ref,
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
    sample_weight: np.ndarray | None = None,
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

    model = train_lgbm_b(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=args.seed,
        device=args.device,
        log_every=max(0, int(args.log_every)),
        sample_weight=sample_weight,
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
        "model": "LGBM-B",
        "sample_weight_col": sw_col or None,
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
            "threshold_tie_ref": float(args.threshold_tie_ref),
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
