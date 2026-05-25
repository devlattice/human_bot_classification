#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace.model.scripts.lgbm import _threshold_sweep, evaluate, load_parquet_pair
from workspace.model.scripts.lgbm_2 import train_lgbm_b_v2


def _metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true).reshape(-1).astype(np.int64)
    p = np.asarray(y_score).reshape(-1).astype(float)
    pred = (p >= float(threshold)).astype(np.int64)
    n_h = int((y == 0).sum())
    n_b = int((y == 1).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    acc = float(np.mean(pred == y)) if len(y) > 0 else 0.0
    human_fpr = float(fp) / float(n_h) if n_h > 0 else 0.0
    bot_recall = float(tp) / float(n_b) if n_b > 0 else 0.0
    return {"human_fpr": human_fpr, "bot_recall": bot_recall, "accuracy": acc}


def _require_optuna():
    try:
        import optuna  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("Optuna is required. Install `optuna`.") from e
    return __import__("optuna")


def _score_val(
    model,
    X_val,
    y_val: np.ndarray,
    *,
    objective: str,
    fpr_cap: float,
    threshold_fixed: float,
) -> tuple[float, dict[str, Any]]:
    y_score = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
    if objective == "bot_recall_at_05":
        m = evaluate(model, X_val, y_val, threshold=threshold_fixed)
        score = float(m["recall"]) - 0.25 * max(0.0, float(m["human_fpr"]) - float(fpr_cap))
        return score, {"metric": m}

    sweep = _threshold_sweep(
        y_true=y_val,
        y_score=y_score,
        target_human_fpr=float(fpr_cap),
        grid_size=1001,
        threshold_tie_ref=0.5,
    )
    br = float(sweep["selected_metrics"]["bot_recall"])
    hf = float(sweep["selected_metrics"]["human_fpr"])
    score = br if sweep["hit_target"] else br - 0.1 * max(0.0, hf - float(fpr_cap))
    return score, {"sweep": sweep}


def _load_holdout_xy(path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path.expanduser().resolve())
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing label")
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing {len(miss)} features, e.g. {miss[:3]}")
    return df[feature_cols], df["label"].to_numpy()


def _eval_domains_fixed_threshold(
    model,
    domain_paths: list[Path],
    feature_cols: list[str],
    *,
    threshold: float,
    fpr_cap: float,
) -> dict[str, Any]:
    per: list[dict[str, Any]] = []
    for p in domain_paths:
        Xd, yd = _load_holdout_xy(p, feature_cols)
        y_score = np.asarray(model.predict_proba(Xd)[:, 1], dtype=float)
        m = _metrics_from_scores(yd, y_score, threshold=threshold)
        per.append(
            {
                "path": str(p.expanduser().resolve()),
                "n": int(len(yd)),
                "human_fpr": float(m["human_fpr"]),
                "bot_recall": float(m["bot_recall"]),
                "accuracy": float(m["accuracy"]),
                "feasible": bool(float(m["human_fpr"]) <= float(fpr_cap) + 1e-12),
            }
        )
    if not per:
        return {
            "per_domain": [],
            "all_feasible": False,
            "worst_bot_recall": 0.0,
            "mean_bot_recall": 0.0,
            "mean_human_fpr": 1.0,
            "mean_accuracy": 0.0,
        }
    brs = [float(x["bot_recall"]) for x in per]
    hfs = [float(x["human_fpr"]) for x in per]
    accs = [float(x["accuracy"]) for x in per]
    return {
        "per_domain": per,
        "all_feasible": all(bool(x["feasible"]) for x in per),
        "worst_bot_recall": float(min(brs)),
        "mean_bot_recall": float(np.mean(brs)),
        "mean_human_fpr": float(np.mean(hfs)),
        "mean_accuracy": float(np.mean(accs)),
    }


def _eval_holdouts(
    model,
    holdouts: list[Path],
    feature_cols: list[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    per: list[dict[str, Any]] = []
    for p in holdouts:
        Xh, yh = _load_holdout_xy(p, feature_cols)
        m = evaluate(model, Xh, yh, threshold=threshold)
        per.append(
            {
                "path": str(p.expanduser().resolve()),
                "n": int(len(yh)),
                "human_fpr": float(m["human_fpr"]),
                "bot_recall": float(m["recall"]),
                "accuracy": float(m["accuracy"]),
                "roc_auc": float(m.get("roc_auc", float("nan"))),
            }
        )
    vals = [float(x["human_fpr"]) for x in per if np.isfinite(x["human_fpr"])]
    return {
        "per_holdout": per,
        "max_human_fpr": max(vals) if vals else 0.0,
        "mean_human_fpr": float(np.mean(vals)) if vals else 0.0,
    }


def suggest_params(trial: Any, n_train_rows: int, regularization: str) -> dict[str, Any]:
    n_train_rows = max(1, int(n_train_rows))
    mcs_upper = max(20, min(1200, int(0.25 * n_train_rows)))
    mcs_lower = max(10, min(100, max(10, int(0.02 * n_train_rows))))
    if mcs_lower > mcs_upper:
        mcs_lower = max(10, mcs_upper // 2)
    if regularization == "strong":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 600, 4000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 28),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", mcs_lower, mcs_upper, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 25.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 50.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 0.35),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 80, 220),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 6000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", mcs_lower, mcs_upper, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 40.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.3),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 80, 250),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna tuning for student LGBM on adapter features.")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory with train.parquet and val.parquet")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    p.add_argument("--lgbm-device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument(
        "--objective",
        choices=("bot_recall_at_05", "max_bot_recall_at_fpr", "multi_objective_generalization_at_05"),
        default="multi_objective_generalization_at_05",
    )
    p.add_argument("--fpr-cap", type=float, default=0.05)
    p.add_argument("--threshold-fixed", type=float, default=0.5)
    p.add_argument(
        "--tune-parquet",
        action="append",
        default=[],
        help="Labeled tune domain parquet(s) used in objective ranking (repeatable).",
    )
    p.add_argument(
        "--test-parquet",
        action="append",
        default=[],
        help="Labeled test domain parquet(s) for final reporting only (repeatable).",
    )
    p.add_argument("--holdout-parquet", action="append", default=[])
    p.add_argument("--holdout-fpr-cap", type=float, default=0.08)
    p.add_argument("--holdout-penalty", type=float, default=0.5)
    p.add_argument("--lgbm-regularization", choices=("strong", "balanced"), default="strong")
    args = p.parse_args()

    optuna = _require_optuna()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, feature_cols = load_parquet_pair(args.data_dir.expanduser().resolve())
    X_train = train_df[feature_cols]
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["label"].to_numpy()
    n_train_rows = int(len(train_df))
    tune_domains = [Path(p).expanduser().resolve() for p in args.tune_parquet]
    test_domains = [Path(p).expanduser().resolve() for p in args.test_parquet]
    holdouts_legacy = [Path(p).expanduser().resolve() for p in args.holdout_parquet]
    if holdouts_legacy and not test_domains:
        test_domains = holdouts_legacy

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    else:
        sampler = optuna.samplers.RandomSampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective_fn(trial: Any) -> float:
        hp = suggest_params(trial, n_train_rows=n_train_rows, regularization=args.lgbm_regularization)
        model = train_lgbm_b_v2(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=int(args.seed) + int(trial.number),
            device=args.lgbm_device,
            log_every=int(args.log_every),
            sample_weight=None,
            **hp,
        )
        val_score, detail = _score_val(
            model,
            X_val,
            y_val,
            objective=args.objective,
            fpr_cap=float(args.fpr_cap),
            threshold_fixed=float(args.threshold_fixed),
        )
        y_score = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
        score_std = float(np.nanstd(y_score))
        score_span = float(np.nanpercentile(y_score, 99) - np.nanpercentile(y_score, 1))
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
        if score_std < 1e-5 or score_span < 1e-4 or best_iter <= 1:
            trial.set_user_attr("degenerate", True)
            return -1.0e9
        trial.set_user_attr("degenerate", False)

        if args.objective == "multi_objective_generalization_at_05":
            domains = [args.data_dir.expanduser().resolve() / "val.parquet", *tune_domains]
            dom_eval = _eval_domains_fixed_threshold(
                model,
                domains,
                feature_cols,
                threshold=float(args.threshold_fixed),
                fpr_cap=float(args.fpr_cap),
            )
            all_feasible = 1.0 if bool(dom_eval["all_feasible"]) else 0.0
            worst_br = float(dom_eval["worst_bot_recall"])
            mean_br = float(dom_eval["mean_bot_recall"])
            mean_hf = float(dom_eval["mean_human_fpr"])
            mean_acc = float(dom_eval["mean_accuracy"])
            rank_key = (all_feasible, worst_br, mean_br, -mean_hf, mean_acc)
            # Scalarization preserving lexicographic priority.
            score = 1.0e8 * all_feasible + 1.0e6 * worst_br + 1.0e4 * mean_br - 1.0e2 * mean_hf + mean_acc
            trial.set_user_attr("rank_key", [float(x) for x in rank_key])
            trial.set_user_attr("tune_eval", dom_eval)
            return float(score)

        if test_domains:
            if args.objective == "max_bot_recall_at_fpr":
                thr = float((detail.get("sweep") or {}).get("selected_threshold", args.threshold_fixed))
            else:
                thr = float(args.threshold_fixed)
            h = _eval_holdouts(model, test_domains, feature_cols, threshold=thr)
            excess = max(0.0, float(h["max_human_fpr"]) - float(args.holdout_fpr_cap))
            penalty = float(args.holdout_penalty) * excess
            trial.set_user_attr("holdout_eval", h)
            trial.set_user_attr("holdout_threshold", float(thr))
            trial.set_user_attr("holdout_penalty", float(penalty))
            return float(val_score) - penalty
        return float(val_score)

    study.optimize(objective_fn, n_trials=int(args.n_trials), show_progress_bar=True)
    best = study.best_trial
    best_params = dict(best.params)

    # Retrain best model once with deterministic seed.
    best_model = train_lgbm_b_v2(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=int(args.seed),
        device=args.lgbm_device,
        log_every=int(args.log_every),
        sample_weight=None,
        **best_params,
    )
    val_score, detail = _score_val(
        best_model,
        X_val,
        y_val,
        objective=args.objective,
        fpr_cap=float(args.fpr_cap),
        threshold_fixed=float(args.threshold_fixed),
    )
    if args.objective == "max_bot_recall_at_fpr":
        selected_threshold = float((detail.get("sweep") or {}).get("selected_threshold", args.threshold_fixed))
    else:
        selected_threshold = float(args.threshold_fixed)
    val_metrics = evaluate(best_model, X_val, y_val, threshold=selected_threshold)
    train_metrics = evaluate(best_model, X_train, y_train, threshold=selected_threshold)

    summary = {
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "n_trials": int(args.n_trials),
        "objective": args.objective,
        "fpr_cap": float(args.fpr_cap),
        "threshold_fixed": float(args.threshold_fixed),
        "tune_parquet": [str(p) for p in tune_domains],
        "test_parquet": [str(p) for p in test_domains],
        "best_value": float(study.best_value),
        "best_params": best_params,
        "selected_threshold": selected_threshold,
        "train_metrics_at_selected_threshold": train_metrics,
        "val_metrics_at_selected_threshold": val_metrics,
        "val_detail": detail,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
    }
    if args.objective == "multi_objective_generalization_at_05":
        domains = [args.data_dir.expanduser().resolve() / "val.parquet", *tune_domains]
        summary["tune_eval"] = _eval_domains_fixed_threshold(
            best_model,
            domains,
            feature_cols,
            threshold=float(args.threshold_fixed),
            fpr_cap=float(args.fpr_cap),
        )
    if test_domains:
        h = _eval_holdouts(best_model, test_domains, feature_cols, threshold=selected_threshold)
        summary["test_eval"] = h
    (out_dir / "optuna_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trials_rows: list[dict[str, Any]] = []
    for t in study.trials:
        row = {
            "number": int(t.number),
            "value": float(t.value) if t.value is not None else float("nan"),
            "state": str(t.state),
            "degenerate": bool(t.user_attrs.get("degenerate", False)),
        }
        if "rank_key" in t.user_attrs:
            row["rank_key"] = t.user_attrs.get("rank_key")
        row.update({k: t.params.get(k) for k in sorted(best_params.keys())})
        trials_rows.append(row)
    pd.DataFrame(trials_rows).to_csv(out_dir / "optuna_trials.csv", index=False)

    try:
        import joblib

        joblib.dump(best_model, out_dir / "lgbm_student.joblib")
    except Exception:
        pass
    best_model.booster_.save_model(str(out_dir / "lgbm_student.txt"))
    (out_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols, "n_features": len(feature_cols)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"best_value": study.best_value, "best_params": best_params}, indent=2))


if __name__ == "__main__":
    main()

