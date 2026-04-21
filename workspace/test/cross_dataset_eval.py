#!/usr/bin/env python3
"""
Cross-dataset evaluation for one trained LightGBM classifier.

Loads one model artifact (`lgbm_classifier.joblib`) and evaluates it on:

- **Dataset directories** — each must contain ``train.parquet`` and ``val.parquet``; or
- **Single tables** — pass ``--eval-parquet path/to/file.parquet`` (repeatable).  
  All rows in that file are scored; results fill **val_*** columns; **train_*** are NaN.

Also reports ``poker44.score.scoring.reward`` (validator-style): uses
``predict_proba[:, 1]`` as continuous scores (FPR / recall from rounded preds;
AP from raw probabilities), same as the live validator window.

Example (repo root):
  PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
    --model workspace/model/artifacts/lgbm/lgbm_classifier.joblib \
    --datasets data/lgbm data/lgbm_xl workspace/datasets/lgbm_large

  PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
    --model workspace/model/artifacts/lgbm/lgbm_classifier.joblib \\
    --eval-parquet workspace/datasets/lgbm_holdout_eval/train.parquet \\
    --out-dir workspace/model/artifacts/cross_eval_holdout_train_only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.score.scoring import reward as subnet_reward


def _load_pair(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"{data_dir}: expected both train.parquet and val.parquet"
        )
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError(f"{data_dir}: missing label column")
    feature_cols = [c for c in train_df.columns if c != "label"]
    if not feature_cols:
        raise ValueError(f"{data_dir}: no feature columns")
    val_feats = [c for c in val_df.columns if c != "label"]
    if set(feature_cols) != set(val_feats):
        raise ValueError(
            f"{data_dir}: train vs val feature columns differ "
            f"(train_only={set(feature_cols) - set(val_feats)!r} "
            f"val_only={set(val_feats) - set(feature_cols)!r})"
        )
    return train_df, val_df, feature_cols


def _load_eval_file(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load one labeled feature table (.parquet or .csv)."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"{path}: expected .parquet or .csv")
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing 'label' column")
    feat = [c for c in df.columns if c != "label"]
    if not feat:
        raise ValueError(f"{path}: no feature columns")
    return df, feat


def _nan_train_row_stub() -> Dict[str, Any]:
    """Placeholder train_* fields when evaluating a single file only."""
    return {
        "train_n": math.nan,
        "train_accuracy": math.nan,
        "train_roc_auc": math.nan,
        "train_log_loss": math.nan,
        "train_human_fpr": math.nan,
        "train_bot_recall": math.nan,
        "train_reward": math.nan,
        "train_reward_fpr": math.nan,
        "train_reward_bot_recall": math.nan,
        "train_reward_ap": math.nan,
        "train_reward_human_safety_penalty": math.nan,
        "train_reward_base_score": math.nan,
    }


def _base_estimator_for_feature_names(model: Any) -> Any:
    """Unwrap meta-estimators (e.g. CalibratedClassifierCV) to the underlying GBDT."""
    steps = getattr(model, "steps", None)
    if isinstance(steps, list) and len(steps) > 0:
        return _base_estimator_for_feature_names(steps[-1][1])
    ccl = getattr(model, "calibrated_classifiers_", None)
    if isinstance(ccl, list) and len(ccl) > 0:
        inner = ccl[0]
        est = getattr(inner, "estimator", None)
        if est is not None:
            return _base_estimator_for_feature_names(est)
    return model


def _model_feature_names(model: Any) -> List[str]:
    """Column order the sklearn LightGBM model was fit with (must match at predict time)."""
    base = _base_estimator_for_feature_names(model)
    names = getattr(base, "feature_name_in_", None)
    if names is not None and len(names) > 0:
        return [str(x) for x in names]
    booster = getattr(base, "booster_", None)
    if booster is not None:
        raw = booster.feature_name()
        if raw:
            return [str(x) for x in raw]
    raise ValueError(
        "Cannot infer training feature names from model (no feature_name_in_ / booster.feature_name)."
    )


def _load_selected_threshold_from_metrics(metrics_json_path: Path) -> Optional[float]:
    """Return ``threshold_selection.selected_threshold`` or None if missing / invalid."""
    try:
        payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"[cross_dataset_eval] warning: could not read metrics JSON {metrics_json_path}: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None
    ts = payload.get("threshold_selection")
    if not isinstance(ts, dict):
        print(
            f"[cross_dataset_eval] warning: {metrics_json_path} has no object "
            f"threshold_selection (cannot load selected_threshold).",
            file=sys.stderr,
            flush=True,
        )
        return None
    raw = ts.get("selected_threshold")
    if raw is None:
        print(
            f"[cross_dataset_eval] warning: {metrics_json_path} missing "
            f"threshold_selection.selected_threshold.",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        print(
            f"[cross_dataset_eval] warning: invalid selected_threshold in metrics JSON: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _align_X(df: pd.DataFrame, model_feats: List[str]) -> pd.DataFrame:
    missing = [c for c in model_feats if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet missing {len(missing)} column(s) required by the model "
            f"(e.g. {missing[:8]!r}…). Rebuild with the same preprocess schema as training."
        )
    return df[model_feats]


def _human_fpr(y_true, y_pred) -> float:
    mask = y_true == 0
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == 1).mean())


def _bot_recall(y_true, y_pred) -> float:
    mask = y_true == 1
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == 1).mean())


def _youden_best_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    grid_size: int = 1001,
) -> Dict[str, float]:
    """
    Pick threshold t in [0,1] maximizing Youden's J = TPR - FPR
    (bot recall minus human FPR). Tie-break: higher t (more conservative on humans).

    y: 0 = human, 1 = bot. proba: P(bot).
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(proba, dtype=float)
    human = y == 0
    bot = y == 1
    if not np.any(human) or not np.any(bot):
        nan = float("nan")
        return {
            "youden_threshold": nan,
            "youden_j": nan,
            "accuracy_youden": nan,
            "human_fpr_youden": nan,
            "bot_recall_youden": nan,
        }
    thresholds = np.linspace(0.0, 1.0, max(2, int(grid_size)))
    best_j = -2.0
    best_t = 0.5
    best_acc = float("nan")
    best_fpr = float("nan")
    best_tpr = float("nan")
    for t in thresholds:
        pred = (s >= t).astype(int)
        tpr = float((pred[bot] == 1).mean())
        fpr = float((pred[human] == 1).mean())
        j = tpr - fpr
        acc = float((pred == y).mean())
        if j > best_j + 1e-15 or (abs(j - best_j) <= 1e-15 and t > best_t):
            best_j = j
            best_t = float(t)
            best_acc = acc
            best_fpr = fpr
            best_tpr = tpr
    return {
        "youden_threshold": best_t,
        "youden_j": float(best_j),
        "accuracy_youden": best_acc,
        "human_fpr_youden": best_fpr,
        "bot_recall_youden": best_tpr,
    }


def _youden_columns_for_split(
    prefix: str,
    y_arr: np.ndarray,
    proba: np.ndarray,
    *,
    do_youden: bool,
    grid_size: int,
) -> Dict[str, float]:
    """Keys: {prefix}_youden_threshold, {prefix}_youden_j, {prefix}_accuracy_at_youden, ..."""
    if not do_youden:
        nan = float("nan")
        return {
            f"{prefix}_youden_threshold": nan,
            f"{prefix}_youden_j": nan,
            f"{prefix}_accuracy_at_youden": nan,
            f"{prefix}_human_fpr_at_youden": nan,
            f"{prefix}_bot_recall_at_youden": nan,
        }
    m = _youden_best_metrics(y_arr, proba, grid_size=max(2, int(grid_size)))
    return {
        f"{prefix}_youden_threshold": m["youden_threshold"],
        f"{prefix}_youden_j": m["youden_j"],
        f"{prefix}_accuracy_at_youden": m["accuracy_youden"],
        f"{prefix}_human_fpr_at_youden": m["human_fpr_youden"],
        f"{prefix}_bot_recall_at_youden": m["bot_recall_youden"],
    }


def _smooth_uncertain_score(s: float, a: float, b: float, gamma: float) -> float:
    """Same uncertain-band transform as neurons/miner.py."""
    if s <= a or s >= b:
        return s
    z = (s - a) / (b - a)
    z_soft = z ** gamma
    return a + (b - a) * z_soft


def _maybe_apply_uncertain_smoothing(
    proba: np.ndarray,
    *,
    uncertain_a: float,
    uncertain_b: float,
    uncertain_gamma: float,
) -> np.ndarray:
    """Apply miner-style uncertain-band smoothing iff params are valid."""
    p = np.asarray(proba, dtype=np.float64)
    if not (0.0 <= uncertain_a < uncertain_b <= 1.0) or uncertain_gamma <= 0.0:
        return p
    out = np.array(
        [
            _smooth_uncertain_score(float(s), uncertain_a, uncertain_b, uncertain_gamma)
            for s in p
        ],
        dtype=np.float64,
    )
    return np.clip(out, 0.0, 1.0)


def _metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    *,
    uncertain_a: float,
    uncertain_b: float,
    uncertain_gamma: float,
) -> Dict[str, float]:
    raw_proba = model.predict_proba(X)[:, 1]
    proba = _maybe_apply_uncertain_smoothing(
        raw_proba,
        uncertain_a=uncertain_a,
        uncertain_b=uncertain_b,
        uncertain_gamma=uncertain_gamma,
    )
    pred = (proba >= threshold).astype(int)
    out: Dict[str, float] = {
        "accuracy": float((pred == y.to_numpy()).mean()),
        "human_fpr": _human_fpr(y.to_numpy(), pred),
        "bot_recall": _bot_recall(y.to_numpy(), pred),
        "n": float(len(y)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y, proba, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def _validator_style_reward(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    uncertain_a: float,
    uncertain_b: float,
    uncertain_gamma: float,
) -> Dict[str, float]:
    """Match ``poker44.score.scoring.reward`` inputs: float scores, int labels 0/1."""
    raw_proba = model.predict_proba(X)[:, 1].astype(np.float64)
    proba = _maybe_apply_uncertain_smoothing(
        raw_proba,
        uncertain_a=uncertain_a,
        uncertain_b=uncertain_b,
        uncertain_gamma=uncertain_gamma,
    )
    y_arr = y.to_numpy(dtype=np.int64)
    rew, res = subnet_reward(proba, y_arr)
    return {
        "reward": float(rew),
        "reward_fpr": float(res["fpr"]),
        "reward_bot_recall": float(res["bot_recall"]),
        "reward_ap": float(res["ap_score"]),
        "reward_human_safety_penalty": float(res["human_safety_penalty"]),
        "reward_base_score": float(res["base_score"]),
    }


def _score_quantiles(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    uncertain_a: float,
    uncertain_b: float,
    uncertain_gamma: float,
) -> Dict[str, float]:
    """Quantiles of predicted bot-probability, overall and per class."""
    raw_p = model.predict_proba(X)[:, 1].astype(np.float64)
    p = _maybe_apply_uncertain_smoothing(
        raw_p,
        uncertain_a=uncertain_a,
        uncertain_b=uncertain_b,
        uncertain_gamma=uncertain_gamma,
    )
    y_arr = y.to_numpy(dtype=np.int64)
    human = p[y_arr == 0]
    bot = p[y_arr == 1]
    out: Dict[str, float] = {
        "score_p01_all": float(np.quantile(p, 0.01)) if len(p) else float("nan"),
        "score_p50_all": float(np.quantile(p, 0.50)) if len(p) else float("nan"),
        "score_p99_all": float(np.quantile(p, 0.99)) if len(p) else float("nan"),
        "score_mean_all": float(np.mean(p)) if len(p) else float("nan"),
    }
    if len(human):
        out.update(
            {
                "score_human_p01": float(np.quantile(human, 0.01)),
                "score_human_p50": float(np.quantile(human, 0.50)),
                "score_human_p99": float(np.quantile(human, 0.99)),
                "score_human_mean": float(np.mean(human)),
            }
        )
    else:
        out.update(
            {
                "score_human_p01": float("nan"),
                "score_human_p50": float("nan"),
                "score_human_p99": float("nan"),
                "score_human_mean": float("nan"),
            }
        )
    if len(bot):
        out.update(
            {
                "score_bot_p01": float(np.quantile(bot, 0.01)),
                "score_bot_p50": float(np.quantile(bot, 0.50)),
                "score_bot_p99": float(np.quantile(bot, 0.99)),
                "score_bot_mean": float(np.mean(bot)),
            }
        )
    else:
        out.update(
            {
                "score_bot_p01": float("nan"),
                "score_bot_p50": float("nan"),
                "score_bot_p99": float("nan"),
                "score_bot_mean": float("nan"),
            }
        )
    return out


def _discover_default_datasets(repo_root: Path) -> List[Path]:
    candidates = [
        repo_root / "data" / "lgbm",
        repo_root / "data" / "lgbm_large",
        repo_root / "data" / "lgbm_xl",
        repo_root / "workspace" / "datasets" / "lgbm",
        repo_root / "workspace" / "datasets" / "lgbm_large",
        repo_root / "workspace" / "datasets" / "lgbm_xl",
    ]
    out: List[Path] = []
    seen = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "train.parquet").is_file() and (c / "val.parquet").is_file():
            out.append(c)
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate one trained LGBM on dataset dirs (train+val parquet) and/or single tables (--eval-parquet)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root / "workspace" / "model" / "artifacts" / "lgbm" / "lgbm_classifier.joblib",
        help="Path to saved sklearn LightGBM model (.joblib)",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="*",
        default=None,
        help="Dataset dirs to evaluate (each needs train.parquet + val.parquet). "
        "Omit together with default discovery if you only pass --eval-parquet.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=("both", "train", "val"),
        default="both",
        help="For --datasets dirs: evaluate both train+val (default), or only one split.",
    )
    parser.add_argument(
        "--eval-parquet",
        type=Path,
        action="append",
        default=None,
        metavar="PATH",
        dest="eval_parquet",
        help="Single .parquet or .csv (label + features). All rows scored; val_* columns set; train_* NaN. "
        "Repeatable. Example: .../lgbm_holdout_eval/train.parquet",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="T",
        help="Decision threshold for primary accuracy/human_fpr columns. "
        "Default: use --selected-threshold if set, else selected_threshold from --metrics-json, else 0.5.",
    )
    parser.add_argument(
        "--use-model-threshold",
        action="store_true",
        help="Set threshold(s) from metrics JSON: fills --threshold and/or --selected-threshold "
        "when those flags were omitted (not passed on the command line).",
    )
    parser.add_argument(
        "--selected-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Second operating point (extra *_at_selected columns). "
        "If omitted, reads threshold_selection.selected_threshold from --metrics-json when present. "
        "If you only pass this flag (no --threshold), primary metrics use this value too.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Path to training metrics.json containing threshold_selection.selected_threshold. "
        "Default: sibling of model artifact.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "workspace" / "model" / "artifacts" / "cross_eval",
        help="Where to save comparison table files.",
    )
    parser.add_argument(
        "--no-youden",
        action="store_true",
        help="Disable per-table Youden threshold columns (default: compute them).",
    )
    parser.add_argument(
        "--youden-grid-size",
        type=int,
        default=1001,
        help="Grid points in [0,1] for Youden threshold search.",
    )
    parser.add_argument(
        "--uncertain-a",
        type=float,
        default=-1.0,
        help="Miner-style uncertain band start `a` (enable only when 0<=a<b<=1 and gamma>0).",
    )
    parser.add_argument(
        "--uncertain-b",
        type=float,
        default=-1.0,
        help="Miner-style uncertain band end `b` (enable only when 0<=a<b<=1 and gamma>0).",
    )
    parser.add_argument(
        "--uncertain-gamma",
        type=float,
        default=1.0,
        help="Miner-style uncertain band power `gamma`.",
    )
    args = parser.parse_args()
    threshold_was_set = "--threshold" in sys.argv
    selected_was_set = "--selected-threshold" in sys.argv

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    try:
        import joblib
        model = joblib.load(model_path)
    except ModuleNotFoundError as e:
        if "lightgbm" in str(e):
            raise ModuleNotFoundError(
                "This script loads `lgbm_classifier.joblib`, which requires `lightgbm` "
                "to be importable in this environment. Install from "
                "`workspace/datasets/requirements-lightgbm.txt` (or offline wheel), then retry."
            ) from e
        raise

    model_feats = _model_feature_names(model)

    eval_files: List[Path] = list(args.eval_parquet or [])
    if args.datasets:
        dataset_dirs: List[Path] = list(args.datasets)
    elif eval_files:
        dataset_dirs = []
    else:
        dataset_dirs = _discover_default_datasets(repo_root)

    if not dataset_dirs and not eval_files:
        raise FileNotFoundError(
            "No inputs: pass --datasets (dirs with train+val parquet), "
            "and/or --eval-parquet /path/to/file.parquet (repeatable), "
            "or rely on default dirs under data/ and workspace/datasets/."
        )

    metrics_json_path = (
        args.metrics_json.expanduser().resolve()
        if args.metrics_json is not None
        else (model_path.parent / "metrics.json")
    )
    t_json: Optional[float] = None
    if metrics_json_path.is_file():
        t_json = _load_selected_threshold_from_metrics(metrics_json_path)
    elif args.metrics_json is not None:
        print(
            f"[cross_dataset_eval] warning: --metrics-json not found: {metrics_json_path}",
            file=sys.stderr,
            flush=True,
        )
    elif not selected_was_set or args.use_model_threshold:
        print(
            f"[cross_dataset_eval] note: default metrics path missing: {metrics_json_path} "
            f"(use --metrics-json if your metrics.json lives elsewhere).",
            file=sys.stderr,
            flush=True,
        )

    selected_threshold: Optional[float] = args.selected_threshold
    if selected_threshold is None:
        selected_threshold = t_json

    if args.use_model_threshold:
        if t_json is None:
            print(
                "[cross_dataset_eval] warning: --use-model-threshold but no valid "
                "selected_threshold in metrics JSON; falling back to other rules.",
                file=sys.stderr,
                flush=True,
            )
        else:
            if not threshold_was_set:
                args.threshold = t_json
            if not selected_was_set:
                selected_threshold = t_json

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif selected_threshold is not None:
        threshold = float(selected_threshold)
    else:
        threshold = 0.5

    print(
        f"[cross_dataset_eval] primary threshold={threshold}  "
        f"selected_threshold={selected_threshold}  "
        f"(metrics_json={metrics_json_path if metrics_json_path.is_file() else 'missing'})",
        file=sys.stderr,
        flush=True,
    )

    do_youden = not bool(args.no_youden)
    youden_grid = max(2, int(args.youden_grid_size))
    uncertain_a = float(args.uncertain_a)
    uncertain_b = float(args.uncertain_b)
    uncertain_gamma = float(args.uncertain_gamma)

    rows: List[Dict[str, Any]] = []
    for d in dataset_dirs:
        data_dir = d.expanduser().resolve()
        train_df, val_df, feat = _load_pair(data_dir)
        extras = set(feat) - set(model_feats)
        if extras:
            print(
                f"[cross_dataset_eval] {data_dir.name}: parquet has {len(extras)} feature column(s) "
                f"not in the model; scoring with the model's {len(model_feats)} training features.",
                file=sys.stderr,
                flush=True,
            )
        eval_train = args.dataset_split in ("both", "train")
        eval_val = args.dataset_split in ("both", "val")
        train_m: Dict[str, float]
        val_m: Dict[str, float]
        train_r: Dict[str, float]
        val_r: Dict[str, float]
        X_train = y_train = X_val = y_val = None
        if eval_train:
            X_train = _align_X(train_df, model_feats)
            y_train = train_df["label"]
            train_m = _metrics(
                model,
                X_train,
                y_train,
                threshold=threshold,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            train_r = _validator_style_reward(
                model,
                X_train,
                y_train,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            train_q = _score_quantiles(
                model,
                X_train,
                y_train,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
        else:
            train_m = {
                "n": math.nan,
                "accuracy": math.nan,
                "roc_auc": math.nan,
                "log_loss": math.nan,
                "human_fpr": math.nan,
                "bot_recall": math.nan,
            }
            train_r = {
                "reward": math.nan,
                "reward_fpr": math.nan,
                "reward_bot_recall": math.nan,
                "reward_ap": math.nan,
                "reward_human_safety_penalty": math.nan,
                "reward_base_score": math.nan,
            }
            train_q = {
                "score_p01_all": math.nan,
                "score_p50_all": math.nan,
                "score_p99_all": math.nan,
                "score_mean_all": math.nan,
                "score_human_p01": math.nan,
                "score_human_p50": math.nan,
                "score_human_p99": math.nan,
                "score_human_mean": math.nan,
                "score_bot_p01": math.nan,
                "score_bot_p50": math.nan,
                "score_bot_p99": math.nan,
                "score_bot_mean": math.nan,
            }
        if eval_val:
            X_val = _align_X(val_df, model_feats)
            y_val = val_df["label"]
            val_m = _metrics(
                model,
                X_val,
                y_val,
                threshold=threshold,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            val_r = _validator_style_reward(
                model,
                X_val,
                y_val,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            val_q = _score_quantiles(
                model,
                X_val,
                y_val,
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
        else:
            val_m = {
                "n": math.nan,
                "accuracy": math.nan,
                "roc_auc": math.nan,
                "log_loss": math.nan,
                "human_fpr": math.nan,
                "bot_recall": math.nan,
            }
            val_r = {
                "reward": math.nan,
                "reward_fpr": math.nan,
                "reward_bot_recall": math.nan,
                "reward_ap": math.nan,
                "reward_human_safety_penalty": math.nan,
                "reward_base_score": math.nan,
            }
            val_q = {
                "score_p01_all": math.nan,
                "score_p50_all": math.nan,
                "score_p99_all": math.nan,
                "score_mean_all": math.nan,
                "score_human_p01": math.nan,
                "score_human_p50": math.nan,
                "score_human_p99": math.nan,
                "score_human_mean": math.nan,
                "score_bot_p01": math.nan,
                "score_bot_p50": math.nan,
                "score_bot_p99": math.nan,
                "score_bot_mean": math.nan,
            }
        if eval_train:
            if do_youden and X_train is not None and y_train is not None:
                proba_tr = _maybe_apply_uncertain_smoothing(
                    model.predict_proba(X_train)[:, 1],
                    uncertain_a=uncertain_a,
                    uncertain_b=uncertain_b,
                    uncertain_gamma=uncertain_gamma,
                )
                train_youden_cols = _youden_columns_for_split(
                    "train", y_train.to_numpy(), proba_tr, do_youden=True, grid_size=youden_grid
                )
            else:
                train_youden_cols = _youden_columns_for_split(
                    "train", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
                )
        else:
            train_youden_cols = _youden_columns_for_split(
                "train", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
            )
        if eval_val:
            if do_youden and X_val is not None and y_val is not None:
                proba_va = _maybe_apply_uncertain_smoothing(
                    model.predict_proba(X_val)[:, 1],
                    uncertain_a=uncertain_a,
                    uncertain_b=uncertain_b,
                    uncertain_gamma=uncertain_gamma,
                )
                val_youden_cols = _youden_columns_for_split(
                    "val", y_val.to_numpy(), proba_va, do_youden=True, grid_size=youden_grid
                )
            else:
                val_youden_cols = _youden_columns_for_split(
                    "val", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
                )
        else:
            val_youden_cols = _youden_columns_for_split(
                "val", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
            )
        row: Dict[str, Any] = {
            "dataset_dir": str(data_dir),
            "threshold_used": float(threshold),
            "uncertain_a": uncertain_a,
            "uncertain_b": uncertain_b,
            "uncertain_gamma": uncertain_gamma,
            "n_features": len(model_feats),
            "n_features_parquet": len(feat),
            "train_n": int(train_m["n"]) if not math.isnan(train_m["n"]) else math.nan,
            "train_accuracy": train_m["accuracy"],
            "train_roc_auc": train_m["roc_auc"],
            "train_log_loss": train_m["log_loss"],
            "train_human_fpr": train_m["human_fpr"],
            "train_bot_recall": train_m["bot_recall"],
            "train_reward": train_r["reward"],
            "train_reward_fpr": train_r["reward_fpr"],
            "train_reward_bot_recall": train_r["reward_bot_recall"],
            "train_reward_ap": train_r["reward_ap"],
            "train_reward_human_safety_penalty": train_r["reward_human_safety_penalty"],
            "train_reward_base_score": train_r["reward_base_score"],
            "train_score_p01_all": train_q["score_p01_all"],
            "train_score_p50_all": train_q["score_p50_all"],
            "train_score_p99_all": train_q["score_p99_all"],
            "train_score_mean_all": train_q["score_mean_all"],
            "train_score_human_p01": train_q["score_human_p01"],
            "train_score_human_p50": train_q["score_human_p50"],
            "train_score_human_p99": train_q["score_human_p99"],
            "train_score_human_mean": train_q["score_human_mean"],
            "train_score_bot_p01": train_q["score_bot_p01"],
            "train_score_bot_p50": train_q["score_bot_p50"],
            "train_score_bot_p99": train_q["score_bot_p99"],
            "train_score_bot_mean": train_q["score_bot_mean"],
            "val_n": int(val_m["n"]) if not math.isnan(val_m["n"]) else math.nan,
            "val_accuracy": val_m["accuracy"],
            "val_roc_auc": val_m["roc_auc"],
            "val_log_loss": val_m["log_loss"],
            "val_human_fpr": val_m["human_fpr"],
            "val_bot_recall": val_m["bot_recall"],
            "val_reward": val_r["reward"],
            "val_reward_fpr": val_r["reward_fpr"],
            "val_reward_bot_recall": val_r["reward_bot_recall"],
            "val_reward_ap": val_r["reward_ap"],
            "val_reward_human_safety_penalty": val_r["reward_human_safety_penalty"],
            "val_reward_base_score": val_r["reward_base_score"],
            "val_score_p01_all": val_q["score_p01_all"],
            "val_score_p50_all": val_q["score_p50_all"],
            "val_score_p99_all": val_q["score_p99_all"],
            "val_score_mean_all": val_q["score_mean_all"],
            "val_score_human_p01": val_q["score_human_p01"],
            "val_score_human_p50": val_q["score_human_p50"],
            "val_score_human_p99": val_q["score_human_p99"],
            "val_score_human_mean": val_q["score_human_mean"],
            "val_score_bot_p01": val_q["score_bot_p01"],
            "val_score_bot_p50": val_q["score_bot_p50"],
            "val_score_bot_p99": val_q["score_bot_p99"],
            "val_score_bot_mean": val_q["score_bot_mean"],
            **train_youden_cols,
            **val_youden_cols,
        }
        if selected_threshold is not None:
            train_sel = (
                _metrics(
                    model,
                    X_train,
                    y_train,
                    threshold=float(selected_threshold),
                    uncertain_a=uncertain_a,
                    uncertain_b=uncertain_b,
                    uncertain_gamma=uncertain_gamma,
                )
                if eval_train and X_train is not None and y_train is not None
                else {"accuracy": math.nan, "human_fpr": math.nan, "bot_recall": math.nan}
            )
            val_sel = (
                _metrics(
                    model,
                    X_val,
                    y_val,
                    threshold=float(selected_threshold),
                    uncertain_a=uncertain_a,
                    uncertain_b=uncertain_b,
                    uncertain_gamma=uncertain_gamma,
                )
                if eval_val and X_val is not None and y_val is not None
                else {"accuracy": math.nan, "human_fpr": math.nan, "bot_recall": math.nan}
            )
            row.update(
                {
                    "selected_threshold": float(selected_threshold),
                    "train_accuracy_at_selected": train_sel["accuracy"],
                    "train_human_fpr_at_selected": train_sel["human_fpr"],
                    "train_bot_recall_at_selected": train_sel["bot_recall"],
                    "val_accuracy_at_selected": val_sel["accuracy"],
                    "val_human_fpr_at_selected": val_sel["human_fpr"],
                    "val_bot_recall_at_selected": val_sel["bot_recall"],
                }
            )
        rows.append(row)

    for fp in eval_files:
        path = fp.expanduser().resolve()
        df, feat = _load_eval_file(path)
        extras = set(feat) - set(model_feats)
        if extras:
            print(
                f"[cross_dataset_eval] {path.name}: parquet has {len(extras)} feature column(s) "
                f"not in the model; scoring with the model's {len(model_feats)} training features.",
                file=sys.stderr,
                flush=True,
            )
        X = _align_X(df, model_feats)
        y = df["label"]
        val_m = _metrics(
            model,
            X,
            y,
            threshold=threshold,
            uncertain_a=uncertain_a,
            uncertain_b=uncertain_b,
            uncertain_gamma=uncertain_gamma,
        )
        val_r = _validator_style_reward(
            model,
            X,
            y,
            uncertain_a=uncertain_a,
            uncertain_b=uncertain_b,
            uncertain_gamma=uncertain_gamma,
        )
        val_q = _score_quantiles(
            model,
            X,
            y,
            uncertain_a=uncertain_a,
            uncertain_b=uncertain_b,
            uncertain_gamma=uncertain_gamma,
        )
        train_youden_cols = _youden_columns_for_split(
            "train", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
        )
        if do_youden:
            proba_eval = _maybe_apply_uncertain_smoothing(
                model.predict_proba(X)[:, 1],
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            val_youden_cols = _youden_columns_for_split(
                "val", y.to_numpy(), proba_eval, do_youden=True, grid_size=youden_grid
            )
        else:
            val_youden_cols = _youden_columns_for_split(
                "val", np.zeros(0), np.zeros(0), do_youden=False, grid_size=0
            )
        row = {
            "dataset_dir": str(path),
            "threshold_used": float(threshold),
            "uncertain_a": uncertain_a,
            "uncertain_b": uncertain_b,
            "uncertain_gamma": uncertain_gamma,
            "n_features": len(model_feats),
            "n_features_parquet": len(feat),
            **_nan_train_row_stub(),
            "val_n": int(val_m["n"]),
            "val_accuracy": val_m["accuracy"],
            "val_roc_auc": val_m["roc_auc"],
            "val_log_loss": val_m["log_loss"],
            "val_human_fpr": val_m["human_fpr"],
            "val_bot_recall": val_m["bot_recall"],
            "val_reward": val_r["reward"],
            "val_reward_fpr": val_r["reward_fpr"],
            "val_reward_bot_recall": val_r["reward_bot_recall"],
            "val_reward_ap": val_r["reward_ap"],
            "val_reward_human_safety_penalty": val_r["reward_human_safety_penalty"],
            "val_reward_base_score": val_r["reward_base_score"],
            "val_score_p01_all": val_q["score_p01_all"],
            "val_score_p50_all": val_q["score_p50_all"],
            "val_score_p99_all": val_q["score_p99_all"],
            "val_score_mean_all": val_q["score_mean_all"],
            "val_score_human_p01": val_q["score_human_p01"],
            "val_score_human_p50": val_q["score_human_p50"],
            "val_score_human_p99": val_q["score_human_p99"],
            "val_score_human_mean": val_q["score_human_mean"],
            "val_score_bot_p01": val_q["score_bot_p01"],
            "val_score_bot_p50": val_q["score_bot_p50"],
            "val_score_bot_p99": val_q["score_bot_p99"],
            "val_score_bot_mean": val_q["score_bot_mean"],
            **train_youden_cols,
            **val_youden_cols,
        }
        if selected_threshold is not None:
            val_sel = _metrics(
                model,
                X,
                y,
                threshold=float(selected_threshold),
                uncertain_a=uncertain_a,
                uncertain_b=uncertain_b,
                uncertain_gamma=uncertain_gamma,
            )
            row.update(
                {
                    "selected_threshold": float(selected_threshold),
                    "train_accuracy_at_selected": math.nan,
                    "train_human_fpr_at_selected": math.nan,
                    "train_bot_recall_at_selected": math.nan,
                    "val_accuracy_at_selected": val_sel["accuracy"],
                    "val_human_fpr_at_selected": val_sel["human_fpr"],
                    "val_bot_recall_at_selected": val_sel["bot_recall"],
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        by=["val_roc_auc", "val_accuracy"], ascending=[False, False]
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_dataset_comparison.csv"
    md_path = out_dir / "cross_dataset_comparison.md"
    table.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Cross-dataset evaluation\n\n")
        try:
            f.write(table.to_markdown(index=False))
        except ImportError:
            f.write("```\n")
            f.write(table.to_string(index=False))
            f.write("\n```\n")
        f.write("\n")
        if do_youden and not table.empty:
            f.write("## Youden J on each table (per-dataset threshold)\n\n")
            f.write(
                "For each scored table, `val_*` row: threshold **t** in `[0,1]` that maximizes "
                "**J = bot_recall(t) − human_FPR(t)** (TPR − FPR). "
                "Tie-break: larger **t**. Grid: `--youden-grid-size` (default 1001). "
                "Single-file evals only populate **val_*** Youden columns.\n\n"
            )
            youden_cols = [
                "dataset_dir",
                "val_n",
                "val_roc_auc",
                "val_youden_threshold",
                "val_youden_j",
                "val_accuracy_at_youden",
                "val_human_fpr_at_youden",
                "val_bot_recall_at_youden",
            ]
            slim = table[[c for c in youden_cols if c in table.columns]]
            try:
                f.write(slim.to_markdown(index=False))
            except ImportError:
                f.write("```\n")
                f.write(slim.to_string(index=False))
                f.write("\n```\n")
            f.write("\n")

    print(table.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
