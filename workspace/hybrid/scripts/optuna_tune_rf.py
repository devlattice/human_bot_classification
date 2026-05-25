"""Optuna search for RandomForest hyperparameters (tabular miner model).

Objective (maximize):
    AP on held-out gold day (last day by default)
    minus penalty if zenodo-test human FPR @0.5 exceeds --fpr-cap

Fits transforms on **training** rows only (no leakage), then evaluates.

Writes JSON consumed by ``train_production_model.py --rf-params-json``:
    { "version": 1, "rf_params": {...}, "best_value", "holdout_date", ... }

Usage:
    python workspace/hybrid/scripts/optuna_tune_rf.py --n-trials 50 \\
        --out-json workspace/hybrid/model_bundle/best_rf_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import train_production_model as tpm  # noqa: E402


def datasets_excluding_gold_dates(
    datasets: dict[str, pd.DataFrame],
    exclude: set[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k, df in datasets.items():
        if k == "gold" and exclude and "date" in df.columns:
            m = ~df["date"].astype(str).isin(exclude)
            out[k] = df.loc[m].copy()
        else:
            out[k] = df.copy()
    return out


def suggest_rf_kw(trial, seed: int) -> dict:
    """Trial suggestions → sklearn kwargs (includes random_state, n_jobs)."""
    md = trial.suggest_categorical("max_depth", [4, 6, 8, 12, -1])
    max_depth = None if md == -1 else int(md)

    mf = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.25, 0.4, 0.55])
    if isinstance(mf, str):
        max_features: str | float = mf
    else:
        max_features = float(mf)

    ms = trial.suggest_float("max_samples", 0.55, 1.0)
    cw = trial.suggest_categorical(
        "class_weight", ["balanced", "balanced_subsample", "none"]
    )
    class_weight = None if cw == "none" else cw

    kw: dict = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
        "max_depth": max_depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "max_features": max_features,
        "random_state": int(seed),
        "n_jobs": -1,
        "class_weight": class_weight,
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 5e-3, log=True),
    }
    if ms < 0.999:
        kw["max_samples"] = float(ms)
    return kw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--holdout-date",
        type=str,
        default=None,
        help="Gold date string to hold out (e.g. 2026-05-08). Default: last date in gold.",
    )
    p.add_argument("--fpr-cap", type=float, default=0.02,
                   help="Zenodo-test FPR @0.5 above this adds penalty.")
    p.add_argument("--fpr-penalty", type=float, default=8.0,
                   help="Multiplier on max(0, zen_fpr - fpr_cap).")
    p.add_argument("--zenodo-test-max-rows", type=int, default=4000,
                   help="Cap rows for FPR eval (speed).")
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///workspace/hybrid/optuna_rf.db",
    )
    p.add_argument("--study-name", type=str, default="poker44_rf")
    return p.parse_args()


def main() -> int:
    try:
        import optuna
    except ImportError:
        print("[error] Install optuna: pip install optuna>=3.6", file=sys.stderr)
        return 2

    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = list(tpm.ROBUST_FEATURES)
    datasets = tpm.load_datasets(feature_cols)
    avail = set(feature_cols)
    for ds in datasets.values():
        avail &= set(ds.columns)
    feature_cols = [f for f in feature_cols if f in avail]

    if "gold" not in datasets:
        print("[error] gold_features.parquet required for holdout evaluation")
        return 1

    gold = datasets["gold"]
    dates = sorted(gold["date"].astype(str).unique())
    holdout = args.holdout_date or dates[-1]
    if holdout not in set(dates):
        print(f"[error] holdout date {holdout} not in gold dates: {dates}")
        return 1

    train_ds = datasets_excluding_gold_dates(datasets, {holdout})
    X_train, y_train, _ = tpm.build_training_data(train_ds, feature_cols, args.seed)
    if len(X_train) < 500:
        print("[error] too few training rows after excluding holdout")
        return 1

    X_t_train, transform_meta = tpm.fit_transform_pipeline(X_train, feature_cols)

    gold_h = gold[gold["date"].astype(str) == holdout]
    if len(gold_h) < 20 or gold_h["label"].nunique() < 2:
        print(f"[warn] holdout {holdout} weak for AP; using second-to-last date")
        if len(dates) < 2:
            return 1
        holdout = dates[-2]
        train_ds = datasets_excluding_gold_dates(datasets, {holdout})
        X_train, y_train, _ = tpm.build_training_data(train_ds, feature_cols, args.seed)
        X_t_train, transform_meta = tpm.fit_transform_pipeline(X_train, feature_cols)
        gold_h = gold[gold["date"].astype(str) == holdout]

    X_val = tpm.apply_transform(
        gold_h[feature_cols].values, feature_cols, transform_meta
    )
    y_val = gold_h["label"].values.astype(int)

    zen_path = tpm.TEST_DIR / "zenodo_test_features.parquet"
    if not zen_path.is_file():
        print("[error] zenodo_test_features.parquet required for FPR guard")
        return 1
    zen = pd.read_parquet(zen_path)
    miss = [c for c in feature_cols if c not in zen.columns]
    if miss:
        print(f"[error] zenodo test missing features: {miss[:5]}")
        return 1
    zen = zen.head(args.zenodo_test_max_rows)
    X_zen = tpm.apply_transform(zen[feature_cols].values, feature_cols, transform_meta)

    def objective(trial: "optuna.Trial") -> float:
        kw = suggest_rf_kw(trial, args.seed + trial.number * 9973)
        rf = RandomForestClassifier(**kw)
        rf.fit(X_t_train, y_train)
        p_val = rf.predict_proba(X_val)[:, 1]
        if np.unique(y_val).size < 2:
            ap = 0.0
        else:
            ap = float(average_precision_score(y_val, p_val))
        p_zen = rf.predict_proba(X_zen)[:, 1]
        zen_fpr = float((p_zen >= 0.5).mean())
        viol = max(0.0, zen_fpr - args.fpr_cap)
        score = ap - args.fpr_penalty * viol
        trial.set_user_attr("ap_holdout", ap)
        trial.set_user_attr("zenodo_fpr_0.5", zen_fpr)
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    # Rebuild from FrozenTrial.params (no Trial.suggest_* on completed trials).
    bp = best_trial.params
    md = bp["max_depth"]
    max_depth = None if int(md) == -1 else int(md)
    mf = bp["max_features"]
    max_features: str | float = (
        mf if isinstance(mf, str) else float(mf)
    )
    ms = float(bp["max_samples"])
    cw = bp["class_weight"]
    class_weight = None if cw == "none" else cw

    rf_params = {
        "n_estimators": int(bp["n_estimators"]),
        "max_depth": max_depth,
        "min_samples_leaf": int(bp["min_samples_leaf"]),
        "min_samples_split": int(bp["min_samples_split"]),
        "max_features": max_features,
        "class_weight": class_weight,
        "ccp_alpha": float(bp["ccp_alpha"]),
    }
    if ms < 0.999:
        rf_params["max_samples"] = ms

    payload = {
        "version": 1,
        "best_value": float(study.best_value),
        "best_trial": best_trial.number,
        "n_trials": len(study.trials),
        "holdout_date": holdout,
        "fpr_cap": args.fpr_cap,
        "fpr_penalty": args.fpr_penalty,
        "objective": "ap_holdout - fpr_penalty * max(0, zenodo_fpr@0.5 - fpr_cap)",
        "best_user_attrs": dict(best_trial.user_attrs),
        "rf_params": rf_params,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[done] best_value={study.best_value:.6f}  holdout={holdout}")
    print(f"  best AP≈{best_trial.user_attrs.get('ap_holdout')}  "
          f"zen FPR≈{best_trial.user_attrs.get('zenodo_fpr_0.5')}")
    print(f"  wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
