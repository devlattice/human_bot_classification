#!/usr/bin/env python3
"""
Optuna hyperparameter search for:
  (A) LightGBM only on fixed embedding parquets (--mode lgbm), or
  (B) Nested search: SSL mask/pretrain params (outer) × LGBM (inner) (--mode ssl_lgbm), or
  (C) Mask/pretrain only with fixed LGBM (--mode ssl_lgbm --freeze-lgbm; ssl_embed_v1 concat LGBM baseline).

GPU usage:
  - LightGBM: use --lgbm-device gpu only if your build has a working GPU backend (often OpenCL).
    Default --lgbm-device auto tries GPU once, then falls back to CPU (recommended on WSL without OpenCL).
  - Masked AE pretrain uses sklearn MLPRegressor (CPU only); see pretrain_masked_ae.py.

Objectives (validation set):
  - bot_recall_at_05: maximize bot recall at threshold 0.5 (validator-style).
  - max_bot_recall_at_fpr: maximize bot recall under human_fpr <= --fpr-cap (threshold sweep).

Holdout guard (optional --holdout-parquet):
  After scoring on val, evaluates holdouts using the **same deployment threshold as val** by default:
  for ``max_bot_recall_at_fpr`` this is the val threshold sweep pick; for ``bot_recall_at_05`` it is
  ``--threshold-fixed``. Subtracts ``holdout_penalty * max(0, max_holdout_human_fpr - holdout_fpr_cap)``
  from the val score so Optuna cannot pick configs that look good on val but fail on OOD slices.

Run from repository root:
  PYTHONPATH=. python workspace/ssl_data/ssl_embed/scripts/tune_ssl_lgbm_optuna.py --help
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

# Repo root (parent of the `workspace/` package directory).
REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from workspace.model.scripts.lgbm import (  # noqa: E402
    _threshold_sweep,
    evaluate,
    load_parquet_pair,
)
from workspace.model.scripts.lgbm_2 import train_lgbm_b_v2  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = REPO_ROOT_FOR_IMPORT

# Set on first successful/failed GPU attempt when --lgbm-device auto (for manifest export).
_LGBM_AUTO_RESOLVED: str | None = None


def _lgbm_device_for_manifest(requested: str) -> str:
    """Effective cpu/gpu string for ssl_embed_ablation_manifest.json."""
    if requested == "auto":
        return _LGBM_AUTO_RESOLVED or "cpu"
    return requested


# Frozen LGBM head: matches ssl_embed_v1/lgbm_out_concat/metrics.json (v1 concat baseline).
# See also workspace/ssl_data/ssl_embed/configs/lgbm_v1_concat_baseline.json
FIXED_LGBM_V1_BASELINE: Dict[str, Any] = {
    "n_estimators": 4000,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 400,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 4.0,
    "reg_lambda": 12.0,
    "min_gain_to_split": 0.05,
    "early_stopping_rounds": 150,
}

# Back-compat alias (older docs referred to "README" block; v1 artifact is the source of truth).
FIXED_LGBM_README_DEFAULTS = FIXED_LGBM_V1_BASELINE

# ssl_embed_v1/ssl_model/ssl_masked_ae.json reference (mask + AE geometry).
SSL_MASK_V1_REFERENCE = {
    "mask_ratio": 0.3,
    "mask_mode": "random",
    "mask_mixed_alpha": 0.3,
    "embed_dim": 32,
    "hidden_dim": 96,
    "max_iter": 80,
}


def suggest_outer_ssl_params(
    trial: Any,
    *,
    mask_weight_json: Path | None,
    min_embed_dim: int,
    ssl_search_space: str,
) -> tuple[float, str, float, int, int, int]:
    """
    Outer Optuna suggestions for masked AE.

    v1_near (default): ranges centered on ssl_embed_v1/ssl_model/ssl_masked_ae.json
    (mask ~0.3, embed 32, hidden 96, max_iter ~80).

    wide: legacy broad search (more risk of OOD-unfriendly configs).
    """
    if mask_weight_json is not None:
        mask_mode = trial.suggest_categorical("mask_mode", ("random", "mixed", "weighted"))
    else:
        mask_mode = "random"
    mixed_alpha = trial.suggest_float("mask_mixed_alpha", 0.1, 0.5) if mask_mode == "mixed" else 0.3

    med = int(min_embed_dim)
    if med >= 64:
        ed_choices = (64,)
    elif med >= 32:
        ed_choices = (32, 64)
    else:
        ed_choices = (16, 32, 64)

    if ssl_search_space == "wide":
        mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.45)
        max_iter = trial.suggest_int("max_iter", 40, 120)
    else:
        mask_ratio = trial.suggest_float("mask_ratio", 0.22, 0.38)
        max_iter = trial.suggest_int("max_iter", 60, 100)
    embed_dim = trial.suggest_categorical("embed_dim", ed_choices)
    hidden_dim = trial.suggest_categorical("hidden_dim", (64, 96, 128))
    return mask_ratio, mask_mode, mixed_alpha, embed_dim, hidden_dim, max_iter


def load_fixed_lgbm_params(path: Path | None) -> tuple[Dict[str, Any], str]:
    """Return (params dict for train_lgbm_b_v2, provenance string)."""
    int_keys = {
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "early_stopping_rounds",
    }
    if path is None:
        return dict(FIXED_LGBM_V1_BASELINE), "ssl_embed_v1_lgbm_out_concat"
    p = path.expanduser().resolve()
    payload = json.loads(p.read_text(encoding="utf-8"))
    lgbm = payload.get("lgbm") if isinstance(payload.get("lgbm"), dict) else payload
    if not isinstance(lgbm, dict):
        raise SystemExit(f"--lgbm-fixed-json {p}: expected object or {{'lgbm': {{...}}}}")
    out = dict(FIXED_LGBM_V1_BASELINE)
    for k, v in lgbm.items():
        if k not in out:
            raise SystemExit(f"--lgbm-fixed-json: unknown key {k!r}")
        out[k] = int(v) if k in int_keys else float(v)
    return out, str(p)


def _require_optuna():
    try:
        import optuna  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Install Optuna: pip install -r workspace/ssl_data/ssl_embed/requirements-tune.txt"
        ) from e
    return __import__("optuna")


def _score_val(
    model,
    X_val,
    y_val: np.ndarray,
    *,
    objective: str,
    fpr_cap: float,
    threshold_fixed: float,
) -> tuple[float, Dict[str, Any]]:
    """Return (scalar to maximize, detail dict)."""
    y_val = np.asarray(y_val, dtype=int)
    y_score = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)

    if objective == "bot_recall_at_05":
        m = evaluate(model, X_val, y_val, threshold=threshold_fixed)
        bot_recall = float(m["recall"])
        human_fpr = float(m["human_fpr"])
        # Penalize high FPR slightly so ties prefer safer models
        score = bot_recall - 0.25 * max(0.0, human_fpr - fpr_cap)
        return score, {"metric": m, "objective": objective}

    if objective == "max_bot_recall_at_fpr":
        sweep = _threshold_sweep(
            y_true=y_val,
            y_score=y_score,
            target_human_fpr=float(fpr_cap),
            grid_size=1001,
            threshold_tie_ref=0.5,
        )
        br = float(sweep["selected_metrics"]["bot_recall"])
        hf = float(sweep["selected_metrics"]["human_fpr"])
        score = br if sweep["hit_target"] else br - 0.1 * max(0.0, hf - fpr_cap)
        return score, {"sweep": sweep, "objective": objective}

    raise ValueError(f"unknown objective {objective!r}")


def _load_holdout_xy(path: Path, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path.expanduser().resolve())
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing `label` column")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} feature columns (e.g. {missing[:3]})")
    X = df[feature_cols]
    y = df["label"].to_numpy()
    return X, y


def _eval_holdouts_fixed_threshold(
    model,
    holdout_paths: list[Path],
    feature_cols: list[str],
    *,
    threshold: float,
) -> Dict[str, Any]:
    per: list[Dict[str, Any]] = []
    for p in holdout_paths:
        Xh, yh = _load_holdout_xy(p, feature_cols)
        m = evaluate(model, Xh, yh, threshold=float(threshold))
        per.append(
            {
                "path": str(p.resolve()),
                "n": int(len(yh)),
                "human_fpr": float(m["human_fpr"]),
                "bot_recall": float(m["recall"]),
                "accuracy": float(m["accuracy"]),
            }
        )
    hfs = [float(x["human_fpr"]) for x in per if np.isfinite(x["human_fpr"])]
    max_hf = max(hfs) if hfs else 0.0
    mean_hf = float(np.mean(hfs)) if hfs else 0.0
    return {"per_holdout": per, "max_human_fpr": max_hf, "mean_human_fpr": mean_hf}


def suggest_lgbm_params(trial: Any, *, seed: int, regularization: str) -> Dict[str, Any]:
    """Hyperparameter ranges for LGBM-B v2 style classifier."""
    if regularization == "strong":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 600, 4000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 28),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 250, 1500, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 25.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 50.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 0.35),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 280),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 800, 6000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 1200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 40.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.3),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 80, 250),
    }


def train_and_eval_lgbm(
    data_dir: Path,
    params: Dict[str, Any],
    *,
    seed: int,
    lgbm_device: str,
    log_every: int,
    objective: str,
    fpr_cap: float,
    threshold_fixed: float,
    holdout_parquets: list[Path] | None,
    holdout_fpr_cap: float,
    holdout_penalty_weight: float,
    holdout_threshold: float,
) -> tuple[float, Dict[str, Any]]:
    global _LGBM_AUTO_RESOLVED
    train_df, val_df, feature_cols = load_parquet_pair(data_dir)
    X_train = train_df[feature_cols]
    y_train = train_df["label"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["label"].to_numpy()

    if lgbm_device == "auto":
        use_device = _LGBM_AUTO_RESOLVED or "gpu"
    else:
        use_device = lgbm_device

    try:
        from lightgbm.basic import LightGBMError
    except Exception:  # pragma: no cover
        LightGBMError = RuntimeError  # type: ignore[misc, assignment]

    def _fit(dev: str):
        return train_lgbm_b_v2(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed,
            device=dev,
            log_every=log_every,
            sample_weight=None,
            **params,
        )

    try:
        model = _fit(use_device)
        if lgbm_device == "auto" and _LGBM_AUTO_RESOLVED is None:
            _LGBM_AUTO_RESOLVED = use_device
    except Exception as e:
        gpu_like = isinstance(e, LightGBMError) or any(
            s in str(e) for s in ("OpenCL", "GPU", "gpu", "CUDA", "cuda", "device")
        )
        if lgbm_device == "auto" and use_device == "gpu" and gpu_like:
            print(
                f"[tune] LightGBM GPU unavailable ({e}); using CPU for remaining trials.",
                file=sys.stderr,
                flush=True,
            )
            _LGBM_AUTO_RESOLVED = "cpu"
            model = _fit("cpu")
        elif lgbm_device == "gpu" and gpu_like:
            raise SystemExit(
                "LightGBM failed in GPU mode:\n"
                f"  {e}\n"
                "Use --lgbm-device cpu, or --lgbm-device auto to fall back to CPU "
                "(common on WSL when OpenCL is not available)."
            ) from e
        else:
            raise

    score, detail = _score_val(
        model,
        X_val,
        y_val,
        objective=objective,
        fpr_cap=fpr_cap,
        threshold_fixed=threshold_fixed,
    )
    detail["train_rows"] = len(train_df)
    detail["val_rows"] = len(val_df)
    detail["n_features"] = len(feature_cols)
    detail["val_score_raw"] = float(score)

    if holdout_parquets:
        if objective == "max_bot_recall_at_fpr":
            sw = detail.get("sweep") or {}
            thr_hold = float(sw.get("selected_threshold", holdout_threshold))
            thr_policy = "val_threshold_sweep_selected"
        else:
            thr_hold = float(threshold_fixed)
            thr_policy = "threshold_fixed_matches_val_objective"
        ho = _eval_holdouts_fixed_threshold(
            model,
            holdout_parquets,
            feature_cols,
            threshold=thr_hold,
        )
        excess = max(0.0, float(ho["max_human_fpr"]) - float(holdout_fpr_cap))
        penalty = float(holdout_penalty_weight) * excess
        combined = float(score) - penalty
        detail["holdout_eval"] = ho
        detail["holdout_fpr_cap"] = float(holdout_fpr_cap)
        detail["holdout_threshold_cli"] = float(holdout_threshold)
        detail["holdout_threshold_applied"] = float(thr_hold)
        detail["holdout_threshold_policy"] = thr_policy
        detail["holdout_penalty"] = penalty
        detail["objective_combined"] = combined
        return combined, detail

    return score, detail


def run_lgbm_study(
    *,
    data_dir: Path,
    n_trials: int,
    seed: int,
    lgbm_device: str,
    log_every: int,
    objective: str,
    fpr_cap: float,
    threshold_fixed: float,
    study_name: str | None,
    storage: str | None,
    sampler: str,
    holdout_parquets: list[Path] | None,
    holdout_fpr_cap: float,
    holdout_penalty_weight: float,
    holdout_threshold: float,
    lgbm_regularization: str,
) -> None:
    optuna = _require_optuna()
    if sampler == "tpe":
        smp = optuna.samplers.TPESampler(seed=seed)
    else:
        smp = optuna.samplers.RandomSampler(seed=seed)

    def objective_fn(trial: Any) -> float:
        hp = suggest_lgbm_params(trial, seed=seed + trial.number, regularization=lgbm_regularization)
        score, _ = train_and_eval_lgbm(
            data_dir,
            hp,
            seed=seed,
            lgbm_device=lgbm_device,
            log_every=log_every,
            objective=objective,
            fpr_cap=fpr_cap,
            threshold_fixed=threshold_fixed,
            holdout_parquets=holdout_parquets,
            holdout_fpr_cap=holdout_fpr_cap,
            holdout_penalty_weight=holdout_penalty_weight,
            holdout_threshold=holdout_threshold,
        )
        return score

    if storage:
        study = optuna.create_study(
            study_name=study_name or "lgbm_ssl_embed",
            storage=storage,
            direction="maximize",
            sampler=smp,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=smp,
        )

    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    print(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))


def _run_subprocess(cmd: list[str]) -> None:
    print("[tune] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_ssl_lgbm_nested(
    *,
    pool_parquet: Path,
    feature_cols_file: Path,
    train_parquet: Path,
    val_parquet: Path,
    holdout_parquets: list[Path] | None,
    mask_weight_json: Path | None,
    work_dir: Path,
    outer_trials: int,
    inner_trials: int,
    seed: int,
    lgbm_device: str,
    log_every: int,
    objective: str,
    fpr_cap: float,
    threshold_fixed: float,
    sampler: str,
    holdout_fpr_cap: float,
    holdout_penalty_weight: float,
    holdout_threshold: float,
    min_embed_dim: int,
    lgbm_regularization: str,
    freeze_lgbm: bool,
    lgbm_fixed_params: Dict[str, Any],
    lgbm_fixed_source: str,
    ssl_search_space: str,
) -> None:
    optuna = _require_optuna()
    if sampler == "tpe":
        smp_outer = optuna.samplers.TPESampler(seed=seed)
    else:
        smp_outer = optuna.samplers.RandomSampler(seed=seed)
    study_outer = optuna.create_study(direction="maximize", sampler=smp_outer)
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if freeze_lgbm:
        print(
            f"[tune] --freeze-lgbm: inner LGBM Optuna disabled; params from {lgbm_fixed_source}",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[tune] outer SSL search space: {ssl_search_space} "
        f"(v1_near = near ssl_embed_v1 ssl_masked_ae.json; wide = legacy broad ranges)",
        file=sys.stderr,
        flush=True,
    )

    pretrain_py = SCRIPTS / "pretrain_masked_ae.py"
    export_py = SCRIPTS / "export_embeddings.py"

    def outer_objective(trial: Any) -> float:
        tid = trial.number
        tdir = work_dir / f"outer_{tid:04d}"
        ssl_dir = tdir / "ssl_model"
        emb_dir = tdir / "embeddings"
        lgbm_data = tdir / "lgbm_data"
        tdir.mkdir(parents=True, exist_ok=True)
        ssl_dir.mkdir(parents=True, exist_ok=True)
        emb_dir.mkdir(parents=True, exist_ok=True)
        lgbm_data.mkdir(parents=True, exist_ok=True)

        mask_ratio, mask_mode, mixed_alpha, embed_dim, hidden_dim, max_iter = suggest_outer_ssl_params(
            trial,
            mask_weight_json=mask_weight_json,
            min_embed_dim=min_embed_dim,
            ssl_search_space=ssl_search_space,
        )

        cmd_pre = [
            sys.executable,
            str(pretrain_py),
            "--pool-parquet",
            str(pool_parquet),
            "--feature-cols-file",
            str(feature_cols_file),
            "--out-dir",
            str(ssl_dir),
            "--mask-ratio",
            str(mask_ratio),
            "--mask-mode",
            str(mask_mode),
            "--mask-mixed-alpha",
            str(mixed_alpha),
            "--embed-dim",
            str(embed_dim),
            "--hidden-dim",
            str(hidden_dim),
            "--max-iter",
            str(max_iter),
            "--seed",
            str(seed + tid),
        ]
        if mask_weight_json is not None and mask_mode in ("mixed", "weighted"):
            cmd_pre.extend(["--mask-weight-json", str(mask_weight_json)])
        _run_subprocess(cmd_pre)

        npz_art = ssl_dir / "ssl_masked_ae.npz"
        cmd_exp = [
            sys.executable,
            str(export_py),
            "--artifact",
            str(npz_art),
            "--in-parquet",
            str(train_parquet),
            "--in-parquet",
            str(val_parquet),
        ]
        for hp in holdout_parquets or []:
            cmd_exp.extend(["--in-parquet", str(hp)])
        cmd_exp.extend(["--out-dir", str(emb_dir)])
        _run_subprocess(cmd_exp)

        tr_base = train_parquet.name
        va_base = val_parquet.name
        shutil.copy2(emb_dir / tr_base, lgbm_data / "train.parquet")
        shutil.copy2(emb_dir / va_base, lgbm_data / "val.parquet")

        holdout_emb_paths: list[Path] | None = None
        if holdout_parquets:
            holdout_emb_paths = []
            for hp in holdout_parquets:
                emb_p = emb_dir / Path(hp).name
                if not emb_p.is_file():
                    raise SystemExit(f"export did not produce holdout embedding: {emb_p}")
                holdout_emb_paths.append(emb_p)

        if freeze_lgbm:
            hp = dict(lgbm_fixed_params)
            score, _ = train_and_eval_lgbm(
                lgbm_data,
                hp,
                seed=seed,
                lgbm_device=lgbm_device,
                log_every=log_every,
                objective=objective,
                fpr_cap=fpr_cap,
                threshold_fixed=threshold_fixed,
                holdout_parquets=holdout_emb_paths,
                holdout_fpr_cap=holdout_fpr_cap,
                holdout_penalty_weight=holdout_penalty_weight,
                holdout_threshold=holdout_threshold,
            )
            meta = {
                "outer_trial": tid,
                "outer_params": trial.params,
                "inner_best_value": float(score),
                "inner_best_params": hp,
                "lgbm_frozen": True,
                "lgbm_fixed_source": lgbm_fixed_source,
            }
            (tdir / "nested_trial_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return float(score)

        optuna_inner = _require_optuna()
        if sampler == "tpe":
            smp_inner = optuna_inner.samplers.TPESampler(seed=seed + 1000 + tid)
        else:
            smp_inner = optuna_inner.samplers.RandomSampler(seed=seed + 1000 + tid)

        def inner_objective(it: Any) -> float:
            hp = suggest_lgbm_params(
                it,
                seed=seed + 10000 + tid * 1000 + it.number,
                regularization=lgbm_regularization,
            )
            score, _ = train_and_eval_lgbm(
                lgbm_data,
                hp,
                seed=seed,
                lgbm_device=lgbm_device,
                log_every=log_every,
                objective=objective,
                fpr_cap=fpr_cap,
                threshold_fixed=threshold_fixed,
                holdout_parquets=holdout_emb_paths,
                holdout_fpr_cap=holdout_fpr_cap,
                holdout_penalty_weight=holdout_penalty_weight,
                holdout_threshold=holdout_threshold,
            )
            return score

        study_inner = optuna_inner.create_study(direction="maximize", sampler=smp_inner)
        study_inner.optimize(inner_objective, n_trials=inner_trials, show_progress_bar=False)
        meta = {
            "outer_trial": tid,
            "outer_params": trial.params,
            "inner_best_value": study_inner.best_value,
            "inner_best_params": study_inner.best_params,
            "lgbm_frozen": False,
        }
        (tdir / "nested_trial_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return float(study_inner.best_value)

    study_outer.optimize(outer_objective, n_trials=outer_trials, show_progress_bar=True)

    best_tid = int(study_outer.best_trial.number)
    nested_summary_path = work_dir / f"outer_{best_tid:04d}" / "nested_trial_summary.json"
    if not nested_summary_path.is_file():
        raise SystemExit(f"missing nested trial summary: {nested_summary_path}")
    nested = json.loads(nested_summary_path.read_text(encoding="utf-8"))

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_by": "tune_ssl_lgbm_optuna.py",
        "mode": "ssl_mask" if freeze_lgbm else "ssl_lgbm",
        "paths": {
            "feature_cols_file": str(feature_cols_file.resolve()),
            "train_parquet": str(train_parquet.resolve()),
            "val_parquet": str(val_parquet.resolve()),
            "pool_parquet": str(pool_parquet.resolve()),
            "mask_weight_json": str(mask_weight_json.resolve()) if mask_weight_json else None,
        },
        "ssl": nested.get("outer_params") or {},
        "ssl_seed": int(seed) + best_tid,
        "lgbm": nested.get("inner_best_params") or {},
        "lgbm_device": _lgbm_device_for_manifest(lgbm_device),
        "lgbm_device_requested": lgbm_device,
        "tuning": {
            "objective": objective,
            "fpr_cap": fpr_cap,
            "threshold_fixed": threshold_fixed,
            "outer_trial": best_tid,
            "outer_best_value": float(study_outer.best_value),
            "inner_best_value": nested.get("inner_best_value"),
            "nested_summary_path": str(nested_summary_path),
            "holdout_parquets_raw": [str(p.resolve()) for p in (holdout_parquets or [])],
            "holdout_fpr_cap": holdout_fpr_cap,
            "holdout_penalty_weight": holdout_penalty_weight,
            "holdout_threshold_cli": holdout_threshold,
            "holdout_threshold_policy": "val_sweep_selected_if_max_bot_recall_at_fpr",
            "min_embed_dim": int(min_embed_dim),
            "lgbm_regularization": lgbm_regularization,
            "freeze_lgbm": bool(freeze_lgbm),
            "lgbm_fixed_source": lgbm_fixed_source if freeze_lgbm else None,
            "ssl_search_space": ssl_search_space,
            "ssl_mask_v1_reference": SSL_MASK_V1_REFERENCE,
        },
    }
    man_path = work_dir / "ssl_embed_ablation_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[tune] wrote {man_path} (use with run_ssl_lgbm_ablation.sh --manifest)")

    out_summary = work_dir / "ssl_lgbm_nested_summary.json"
    out_summary.write_text(
        json.dumps(
            {
                "best_value": study_outer.best_value,
                "best_params": study_outer.best_params,
                "ablation_manifest": str(man_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[tune] wrote {out_summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna: LGBM and/or SSL mask + LGBM search.")
    p.add_argument(
        "--mode",
        choices=("lgbm", "ssl_lgbm"),
        default="lgbm",
        help="lgbm: search LGBM only. ssl_lgbm: outer SSL mask, inner LGBM (expensive).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with train.parquet + val.parquet (embedding features). Required for --mode lgbm.",
    )
    p.add_argument("--n-trials", type=int, default=40, help="Trials for lgbm mode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lgbm-device",
        choices=("cpu", "gpu", "auto"),
        default="auto",
        help="LightGBM device: cpu, gpu (OpenCL/CUDA GPU build), or auto (try GPU, then CPU). "
        "Default: auto. On WSL without OpenCL, gpu often errors with 'No OpenCL device found'.",
    )
    p.add_argument("--log-every", type=int, default=0, help="LGBM log period; 0 = quiet.")
    p.add_argument(
        "--objective",
        choices=("bot_recall_at_05", "max_bot_recall_at_fpr"),
        default="max_bot_recall_at_fpr",
        help="Validation objective to maximize.",
    )
    p.add_argument(
        "--fpr-cap",
        type=float,
        default=0.05,
        help="For max_bot_recall_at_fpr: target max human FPR in threshold sweep.",
    )
    p.add_argument("--threshold-fixed", type=float, default=0.5, help="For bot_recall_at_05.")
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None, help="Optuna RDB storage URL, e.g. sqlite:///study.db")
    p.add_argument("--sampler", choices=("random", "tpe"), default="tpe")

    # Nested SSL + LGBM
    p.add_argument("--pool-parquet", type=Path, default=None)
    p.add_argument("--feature-cols-file", type=Path, default=None)
    p.add_argument("--train-parquet", type=Path, default=None)
    p.add_argument("--val-parquet", type=Path, default=None)
    p.add_argument("--mask-weight-json", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--outer-trials", type=int, default=5)
    p.add_argument("--inner-trials", type=int, default=15)

    p.add_argument(
        "--holdout-parquet",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help="Repeatable. Labeled parquet(s) with same raw features as train (exported + scored during tuning). "
        "Penalty uses the same threshold as val: val sweep pick when --objective max_bot_recall_at_fpr, else "
        "--threshold-fixed. --holdout-threshold is only a fallback if sweep metadata is missing.",
    )
    p.add_argument(
        "--holdout-fpr-cap",
        type=float,
        default=0.15,
        help="Allowed max human FPR on worst holdout (at the val-aligned threshold above) before penalty (default: 0.15).",
    )
    p.add_argument(
        "--holdout-penalty",
        type=float,
        default=8.0,
        help="Penalty per unit FPR above cap on worst holdout (default: 8). Increase if val still wins but holdouts fail.",
    )
    p.add_argument(
        "--holdout-threshold",
        type=float,
        default=0.5,
        help="Fallback holdout threshold only; normally overridden by val sweep (max_bot_recall_at_fpr) or "
        "--threshold-fixed (bot_recall_at_05). Default 0.5.",
    )
    p.add_argument(
        "--min-embed-dim",
        type=int,
        default=32,
        help="ssl_lgbm: minimum embed_dim in outer search (default 32 = ssl_embed_v1-style; 16=all of 16,32,64).",
    )
    p.add_argument(
        "--ssl-search-space",
        choices=("v1_near", "wide"),
        default="v1_near",
        help="ssl_lgbm outer trial ranges: v1_near (default) around ssl_embed_v1 masked AE; wide = legacy broad search.",
    )
    p.add_argument(
        "--lgbm-regularization",
        choices=("default", "strong"),
        default="default",
        help="Narrow LGBM search toward stronger regularization (smaller trees, more min_child_samples).",
    )
    p.add_argument(
        "--freeze-lgbm",
        action="store_true",
        help="ssl_lgbm only: search SSL/mask (outer) only; train one LGBM per trial with fixed hparams "
        "(ssl_embed_v1 lgbm_out_concat baseline or --lgbm-fixed-json). Ignores --inner-trials.",
    )
    p.add_argument(
        "--lgbm-fixed-json",
        type=Path,
        default=None,
        help="With --freeze-lgbm: JSON with lgbm keys (same as ssl_embed_ablation_manifest / lgbm_2 --hparams-json). "
        "Omit to use ssl_embed_v1 concat baseline (see workspace/ssl_data/ssl_embed/configs/lgbm_v1_concat_baseline.json).",
    )

    return p.parse_args()


def main() -> int:
    global _LGBM_AUTO_RESOLVED
    _LGBM_AUTO_RESOLVED = None
    args = parse_args()
    _require_optuna()

    holdouts_lgbm: list[Path] | None = None
    if args.holdout_parquet:
        holdouts_lgbm = [Path(p).expanduser().resolve() for p in args.holdout_parquet]

    if args.mode == "lgbm":
        if args.data_dir is None:
            raise SystemExit("--data-dir is required for --mode lgbm")
        run_lgbm_study(
            data_dir=args.data_dir.expanduser().resolve(),
            n_trials=int(args.n_trials),
            seed=int(args.seed),
            lgbm_device=args.lgbm_device,
            log_every=max(0, int(args.log_every)),
            objective=args.objective,
            fpr_cap=float(args.fpr_cap),
            threshold_fixed=float(args.threshold_fixed),
            study_name=args.study_name,
            storage=args.storage,
            sampler=args.sampler,
            holdout_parquets=holdouts_lgbm,
            holdout_fpr_cap=float(args.holdout_fpr_cap),
            holdout_penalty_weight=float(args.holdout_penalty),
            holdout_threshold=float(args.holdout_threshold),
            lgbm_regularization=args.lgbm_regularization,
        )
        return 0

    # ssl_lgbm
    req = [args.pool_parquet, args.feature_cols_file, args.train_parquet, args.val_parquet, args.work_dir]
    if any(x is None for x in req):
        raise SystemExit(
            "--mode ssl_lgbm requires --pool-parquet --feature-cols-file --train-parquet "
            "--val-parquet --work-dir"
        )
    lgbm_fixed_params: Dict[str, Any] = {}
    lgbm_fixed_src = ""
    if args.freeze_lgbm:
        lgbm_fixed_params, lgbm_fixed_src = load_fixed_lgbm_params(args.lgbm_fixed_json)
        print(
            "[tune] --inner-trials is ignored when --freeze-lgbm is set (one LGBM fit per outer trial).",
            file=sys.stderr,
            flush=True,
        )
    run_ssl_lgbm_nested(
        pool_parquet=args.pool_parquet.expanduser().resolve(),
        feature_cols_file=args.feature_cols_file.expanduser().resolve(),
        train_parquet=args.train_parquet.expanduser().resolve(),
        val_parquet=args.val_parquet.expanduser().resolve(),
        holdout_parquets=holdouts_lgbm,
        mask_weight_json=args.mask_weight_json.expanduser().resolve() if args.mask_weight_json else None,
        work_dir=args.work_dir.expanduser().resolve(),
        outer_trials=int(args.outer_trials),
        inner_trials=int(args.inner_trials),
        seed=int(args.seed),
        lgbm_device=args.lgbm_device,
        log_every=max(0, int(args.log_every)),
        objective=args.objective,
        fpr_cap=float(args.fpr_cap),
        threshold_fixed=float(args.threshold_fixed),
        sampler=args.sampler,
        holdout_fpr_cap=float(args.holdout_fpr_cap),
        holdout_penalty_weight=float(args.holdout_penalty),
        holdout_threshold=float(args.holdout_threshold),
        min_embed_dim=int(args.min_embed_dim),
        lgbm_regularization=args.lgbm_regularization,
        freeze_lgbm=bool(args.freeze_lgbm),
        lgbm_fixed_params=lgbm_fixed_params,
        lgbm_fixed_source=lgbm_fixed_src,
        ssl_search_space=args.ssl_search_space,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
