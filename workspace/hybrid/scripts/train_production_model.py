"""Train production Miner-B model using mixed data + RobustScaler.

Combines:
  - Gold data (latest validator evaluation chunks)
  - Zenodo + public human data (diverse human behavior)
  - Full-spectrum generated bots (covering entire parameter space)

Outputs a deployment bundle:
  - model.joblib              (RandomForest classifier)
  - feature_cols.json         (ordered feature list)
  - transform_meta.json       (clip + log1p + robust scale params)
  - retrain_summary.json      (metrics, threshold, date)

Usage:
    python workspace/hybrid/scripts/train_production_model.py [--output-dir workspace/hybrid/model_bundle]

RandomForest tuning (optional; defaults match previous hard-coded values):
    --rf-n-estimators, --rf-max-depth, --rf-min-samples-leaf, --rf-min-samples-split,
    --rf-max-features, --rf-max-samples, --rf-class-weight, --rf-ccp-alpha
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train"
TEST_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test"

GOLD_PATH = TRAIN_DIR / "gold_features.parquet"
MAY8_GOLD_TEST_PATH = TEST_DIR / "may8_gold_test_features.parquet"
MAY8_DATE = "2026-05-08"
ZENODO_PATH = TRAIN_DIR / "zenodo_features.parquet"
PUBLIC_PATH = TRAIN_DIR / "public_features.parquet"
FULL_SPECTRUM_PATH = TRAIN_DIR / "full_spectrum_bot_features.parquet"
GEN_BOT_PATH = TRAIN_DIR / "generated_bot_features.parquet"
CAL_BOT_PATH = TRAIN_DIR / "calibrated_bot_features.parquet"
ACPC_BOT_PATH = TRAIN_DIR / "acpc_bot_features.parquet"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "workspace" / "hybrid" / "model_bundle"

ROBUST_FEATURES = [
    "action_entropy_p50", "bet_minus_fold_mean", "bet_ratio_p50",
    "mean_pot_after_p50", "bet_ratio_mean", "pot_after_over_stack_mean",
    "mean_pot_after_mean", "aggression_factor_max", "pot_growth_std",
    "action_entropy_p90", "bet_ratio_p90", "bet_size_mean_std",
    "std_norm_bb_std", "mean_pot_after_p90", "bet_ratio_std",
    "mean_pot_after_std", "std_pot_after_std", "aggression_factor_mean",
    "aggression_factor_std", "bet_size_mean_p90", "bet_ratio_max",
    "mean_norm_bb_p90", "fold_ratio_p10", "bet_size_max_std",
    "max_norm_bb_std", "pot_growth_mean", "bet_size_mean_mean",
    "action_entropy_max", "max_consecutive_max", "preflop_action_density_std",
    "p6p_std", "aggression_factor_p90", "p5_max", "end_turn_max",
    "call_ratio_p90", "bet_size_mean_p50", "bet_size_mean_max",
    "mean_pot_after_p10", "p6p_max", "n_players_max",
    "fold_position_mean_std", "end_preflop_std", "raise_minus_call_mean",
    "end_flop_max", "end_turn_std", "bet_size_std_max", "end_river_max",
    "n_streets_max", "unique_actors_ratio_max", "end_flop_std",
    "n_streets_std", "end_river_std", "check_ratio_p90", "fold_position_mean_mean",
]

HUMAN_SAMPLE_CAP = 5000
BOT_SAMPLE_CAP = 8000
PUBLIC_OVERSAMPLE = 7


def add_rf_arguments(ap: argparse.ArgumentParser) -> None:
    """Add RandomForest hyperparameter CLI flags to an ArgumentParser."""
    g = ap.add_argument_group("RandomForest hyperparameters")
    g.add_argument("--rf-n-estimators", type=int, default=300)
    g.add_argument(
        "--rf-max-depth",
        type=str,
        default="6",
        help="Max tree depth (integer), or 'none' for unlimited.",
    )
    g.add_argument("--rf-min-samples-leaf", type=int, default=15)
    g.add_argument("--rf-min-samples-split", type=int, default=2)
    g.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt",
        help="sqrt | log2 | none (use all features) | float in (0,1] e.g. 0.35",
    )
    g.add_argument(
        "--rf-max-samples",
        type=float,
        default=1.0,
        help="Bootstrap row fraction per tree; 1.0 = full sample (sklearn default).",
    )
    g.add_argument(
        "--rf-class-weight",
        type=str,
        default="balanced",
        choices=("balanced", "balanced_subsample", "none"),
    )
    g.add_argument(
        "--rf-ccp-alpha",
        type=float,
        default=0.0,
        help="Minimal cost-complexity pruning (0 = disabled).",
    )


def rf_kwargs_from_namespace(ns: argparse.Namespace, seed: int) -> dict:
    """Build kwargs for sklearn RandomForestClassifier from argparse namespace."""
    md = str(ns.rf_max_depth).strip().lower()
    if md in ("none", "full", "null", ""):
        max_depth = None
    else:
        max_depth = int(ns.rf_max_depth)

    mfs = str(ns.rf_max_features).strip().lower()
    if mfs in ("sqrt", "log2"):
        max_features: str | float | None = mfs
    elif mfs in ("none", "null", "all"):
        max_features = None
    else:
        max_features = float(ns.rf_max_features)
        if not (0.0 < max_features <= 1.0):
            raise ValueError(
                "--rf-max-features must be sqrt, log2, none, or a float in (0, 1]"
            )

    cw = None if ns.rf_class_weight == "none" else ns.rf_class_weight

    kw: dict = {
        "n_estimators": int(ns.rf_n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(ns.rf_min_samples_leaf),
        "min_samples_split": int(ns.rf_min_samples_split),
        "max_features": max_features,
        "random_state": int(seed),
        "n_jobs": -1,
        "class_weight": cw,
        "ccp_alpha": float(ns.rf_ccp_alpha),
    }
    ms = float(ns.rf_max_samples)
    if 0.0 < ms < 1.0:
        kw["max_samples"] = ms
    return kw


def rf_params_for_summary(rf_kwargs: dict) -> dict:
    """JSON-serializable copy of RF kwargs (for retrain_summary.json)."""
    out = {}
    for k, v in rf_kwargs.items():
        if k == "n_jobs":
            continue
        out[k] = v
    return out


# Keys allowed in ``best_rf_params.json`` from ``optuna_tune_rf.py``.
_RF_PATCH_KEYS = frozenset({
    "n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
    "max_features", "max_samples", "class_weight", "ccp_alpha",
})


def load_rf_params_json(path: Path) -> dict:
    """Load patch dict; supports ``{\"rf_params\": {...}}`` or flat dict."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["rf_params"] if isinstance(raw.get("rf_params"), dict) else raw


def apply_rf_params_patch(base_kwargs: dict, patch: dict) -> dict:
    """Merge Optuna / JSON patch into full sklearn kwargs (random_state / n_jobs preserved)."""
    out = dict(base_kwargs)
    for k, v in patch.items():
        if k not in _RF_PATCH_KEYS:
            continue
        if k == "max_depth":
            if v is None or (isinstance(v, str) and str(v).strip().lower() in ("none", "null", "")):
                out[k] = None
            else:
                out[k] = int(v)
        elif k == "max_features":
            if isinstance(v, str):
                vl = v.strip().lower()
                if vl in ("sqrt", "log2"):
                    out[k] = vl
                elif vl in ("none", "null", "all"):
                    out[k] = None
                else:
                    out[k] = float(v)
            elif v is None:
                out[k] = None
            else:
                out[k] = float(v)
        elif k == "class_weight":
            if v is None or (isinstance(v, str) and str(v).strip().lower() in ("none", "null")):
                out[k] = None
            else:
                out[k] = str(v)
        elif k == "max_samples":
            if v is None or float(v) >= 0.999:
                out.pop("max_samples", None)
            else:
                out[k] = float(v)
        elif k == "ccp_alpha":
            out[k] = float(v)
        elif k in ("n_estimators", "min_samples_leaf", "min_samples_split"):
            out[k] = int(v)
    return out


def load_may8_gold_test() -> pd.DataFrame | None:
    """Rotation hold-out gold (May-8); lives under dataset/test after split."""
    if not MAY8_GOLD_TEST_PATH.is_file():
        return None
    return pd.read_parquet(MAY8_GOLD_TEST_PATH)


def load_datasets(feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Load all available datasets, filtering to common features."""
    datasets = {}

    for name, path in [
        ("gold", GOLD_PATH),
        ("zenodo", ZENODO_PATH),
        ("public", PUBLIC_PATH),
        ("full_spectrum", FULL_SPECTRUM_PATH),
        ("gen_bot", GEN_BOT_PATH),
        ("cal_bot", CAL_BOT_PATH),
        ("acpc_bot", ACPC_BOT_PATH),
    ]:
        if path.is_file():
            df = pd.read_parquet(path)
            missing = [f for f in feature_cols if f not in df.columns]
            if missing:
                print(f"  WARNING: {name} missing {len(missing)} features, skipping: {missing[:5]}")
                continue
            datasets[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
        else:
            print(f"  SKIP {name}: {path} not found")

    return datasets


def build_training_data(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build balanced mixed training set."""
    rng = np.random.RandomState(seed)
    parts_human = []
    parts_bot = []
    meta = {"sources": {}}

    # Gold: all of it (both human and bot)
    if "gold" in datasets:
        gold = datasets["gold"]
        parts_human.append(gold[gold["label"] == 0][feature_cols])
        parts_bot.append(gold[gold["label"] == 1][feature_cols])
        meta["sources"]["gold_human"] = int((gold["label"] == 0).sum())
        meta["sources"]["gold_bot"] = int((gold["label"] == 1).sum())

    # Zenodo humans: subsample to cap
    if "zenodo" in datasets:
        zen = datasets["zenodo"]
        n = min(len(zen), HUMAN_SAMPLE_CAP)
        parts_human.append(zen.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["zenodo_human"] = n

    # Public humans: oversample then cap
    if "public" in datasets:
        pub = datasets["public"]
        pub_up = pd.concat([pub] * PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), HUMAN_SAMPLE_CAP)
        parts_human.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["public_human"] = n

    # Full spectrum bots: primary bot source
    if "full_spectrum" in datasets:
        fs = datasets["full_spectrum"]
        n = min(len(fs), BOT_SAMPLE_CAP)
        parts_bot.append(fs.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["full_spectrum_bot"] = n

    # Generated bots: supplement
    if "gen_bot" in datasets:
        gb = datasets["gen_bot"]
        n = min(len(gb), BOT_SAMPLE_CAP // 2)
        parts_bot.append(gb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["gen_bot"] = n

    # Calibrated bots: supplement
    if "cal_bot" in datasets:
        cb = datasets["cal_bot"]
        n = min(len(cb), BOT_SAMPLE_CAP // 2)
        parts_bot.append(cb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["cal_bot"] = n

    # ACPC competition bots: sophisticated AI bots
    if "acpc_bot" in datasets:
        acpc = datasets["acpc_bot"]
        n = min(len(acpc), BOT_SAMPLE_CAP)
        parts_bot.append(acpc.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["acpc_bot"] = n

    human_df = pd.concat(parts_human, ignore_index=True) if parts_human else pd.DataFrame()
    bot_df = pd.concat(parts_bot, ignore_index=True) if parts_bot else pd.DataFrame()

    X = pd.concat([human_df, bot_df], ignore_index=True).values
    y = np.concatenate([np.zeros(len(human_df)), np.ones(len(bot_df))])
    meta["total_human"] = int(len(human_df))
    meta["total_bot"] = int(len(bot_df))
    meta["total"] = int(len(X))

    return X, y, meta


def fit_transform_pipeline(X: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, dict]:
    """Apply quantile clipping + log1p + robust scaling, return transformed X and meta."""
    df = pd.DataFrame(X, columns=feature_cols)

    log1p_features = [f for f in feature_cols
                      if any(k in f.lower() for k in ("pot", "stack", "norm_bb", "bet_size"))]

    clip_bounds = {}
    for c in feature_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        lo = float(np.quantile(s, 0.01))
        hi = float(np.quantile(s, 0.99))
        clip_bounds[c] = {"low": lo, "high": hi}
        df[c] = df[c].clip(lower=lo, upper=hi)

    for c in log1p_features:
        if c in df.columns:
            s = df[c].astype(float)
            df[c] = np.sign(s) * np.log1p(np.abs(s))

    robust_stats = {}
    for c in feature_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, median, q3 = float(np.quantile(s, 0.25)), float(np.median(s)), float(np.quantile(s, 0.75))
        iqr = q3 - q1
        if iqr == 0.0:
            iqr = 1.0
        robust_stats[c] = {"median": median, "iqr": iqr}
        df[c] = (df[c] - median) / iqr

    fill_medians = {}
    for c in feature_cols:
        med = df[c].median()
        if np.isfinite(med):
            fill_medians[c] = float(med)
    df = df.fillna(fill_medians).fillna(0.0)

    transform_meta = {
        "clip": {"enabled": True, "q_low": 0.01, "q_high": 0.99},
        "clip_bounds": clip_bounds,
        "clip_features_count": len(clip_bounds),
        "log1p": {"enabled": True},
        "log1p_selected_features": log1p_features,
        "log1p_selected_count": len(log1p_features),
        "robust_scale": {"enabled": True, "scaled_clip_abs": 0.0},
        "robust_scale_stats": robust_stats,
        "robust_scale_features_count": len(robust_stats),
        "fillna": {
            "method": "train_median_then_zero",
            "median_features_count": len(fill_medians),
            "medians": fill_medians,
        },
    }

    return df.values, transform_meta


def apply_transform(X: np.ndarray, feature_cols: list[str], meta: dict) -> np.ndarray:
    """Apply saved transform to new data."""
    df = pd.DataFrame(X, columns=feature_cols)

    clip_bounds = meta.get("clip_bounds", {})
    for c, b in clip_bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=b["low"], upper=b["high"])

    log1p_cols = meta.get("log1p_selected_features", [])
    for c in log1p_cols:
        if c in df.columns:
            s = df[c].astype(float)
            df[c] = np.sign(s) * np.log1p(np.abs(s))

    robust_stats = meta.get("robust_scale_stats", {})
    for c, st in robust_stats.items():
        if c in df.columns:
            df[c] = (df[c] - st["median"]) / st["iqr"]

    medians = meta.get("fillna", {}).get("medians", {})
    for c, m in medians.items():
        if c in df.columns:
            df[c] = df[c].fillna(m)
    df = df.fillna(0.0)

    return df.values


def evaluate_gold_loocv(
    gold: pd.DataFrame,
    external_human: np.ndarray,
    external_bot: np.ndarray,
    feature_cols: list[str],
    transform_meta: dict,
    rf_kwargs: dict,
) -> list[dict]:
    """Leave-one-day-out evaluation on gold data."""
    dates = sorted(gold["date"].unique())
    results = []

    for test_date in dates:
        test = gold[gold["date"] == test_date]
        train_gold = gold[gold["date"] != test_date]

        train_X = np.vstack([
            train_gold[feature_cols].values,
            external_human,
            external_bot,
        ])
        train_y = np.concatenate([
            train_gold["label"].values,
            np.zeros(len(external_human)),
            np.ones(len(external_bot)),
        ])

        X_t = apply_transform(train_X, feature_cols, transform_meta)
        test_X_t = apply_transform(test[feature_cols].values, feature_cols, transform_meta)

        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(X_t, train_y)
        proba = rf.predict_proba(test_X_t)[:, 1]

        auc = roc_auc_score(test["label"].values, proba) if len(test["label"].unique()) > 1 else float("nan")
        acc = accuracy_score(test["label"].values, (proba >= 0.5).astype(int))
        n_bot = int((test["label"] == 1).sum())
        bot_detect = int(((proba >= 0.5) & (test["label"].values == 1)).sum()) / max(n_bot, 1)

        results.append({
            "date": test_date,
            "auc": round(float(auc), 6),
            "accuracy": round(float(acc), 6),
            "bot_detect_rate": round(float(bot_detect), 4),
            "n_samples": len(test),
        })

    return results


def find_optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Find threshold that maximizes accuracy."""
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        acc = accuracy_score(y_true, (proba >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return round(float(best_t), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rf-params-json",
        type=Path,
        default=None,
        help="JSON from optuna_tune_rf.py (rf_params); overrides RF CLI flags for same keys.",
    )
    add_rf_arguments(ap)
    args = ap.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

    feature_cols = list(ROBUST_FEATURES)

    print("Loading datasets...")
    datasets = load_datasets(feature_cols)

    available_feats = set(feature_cols)
    for ds in datasets.values():
        available_feats &= set(ds.columns)
    feature_cols = [f for f in feature_cols if f in available_feats]
    print(f"\nUsing {len(feature_cols)} features common to all datasets")

    print("\nBuilding training data...")
    X_raw, y, data_meta = build_training_data(datasets, feature_cols, args.seed)
    print(f"  Total: {data_meta['total']} ({data_meta['total_human']} human, {data_meta['total_bot']} bot)")

    print("\nFitting transforms (clip + log1p + robust scale)...")
    X_transformed, transform_meta = fit_transform_pipeline(X_raw, feature_cols)
    print(f"  Clipped: {transform_meta['clip_features_count']} features")
    print(f"  Log1p: {transform_meta['log1p_selected_count']} features")
    print(f"  Robust scaled: {transform_meta['robust_scale_features_count']} features")

    rf_kwargs = rf_kwargs_from_namespace(args, args.seed)
    if args.rf_params_json is not None and args.rf_params_json.is_file():
        patch = load_rf_params_json(args.rf_params_json)
        rf_kwargs = apply_rf_params_patch(rf_kwargs, patch)
        print(f"\n  RF patch from {args.rf_params_json}")
    print("\nTraining RandomForest...")
    print(f"  RF params: {rf_params_for_summary(rf_kwargs)}")
    t0 = time.time()
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X_transformed, y)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    train_proba = rf.predict_proba(X_transformed)[:, 1]
    train_auc = roc_auc_score(y, train_proba)
    train_acc = accuracy_score(y, (train_proba >= 0.5).astype(int))
    optimal_threshold = find_optimal_threshold(y, train_proba)
    print(f"  Train AUC: {train_auc:.6f}")
    print(f"  Train Acc: {train_acc:.6f}")
    print(f"  Optimal threshold: {optimal_threshold}")

    cv_results: list = []
    mean_auc = min_auc = None
    # Leave-one-day-out on gold
    if "gold" in datasets:
        print("\nLeave-one-day-out validation on gold...")
        ext_human_parts = []
        ext_bot_parts = []
        rng = np.random.RandomState(args.seed)
        if "zenodo" in datasets:
            ext_human_parts.append(datasets["zenodo"].sample(n=min(len(datasets["zenodo"]), HUMAN_SAMPLE_CAP), random_state=rng)[feature_cols].values)
        if "public" in datasets:
            pub_up = pd.concat([datasets["public"]] * PUBLIC_OVERSAMPLE, ignore_index=True)
            ext_human_parts.append(pub_up.sample(n=min(len(pub_up), HUMAN_SAMPLE_CAP), random_state=rng)[feature_cols].values)
        if "full_spectrum" in datasets:
            ext_bot_parts.append(datasets["full_spectrum"].sample(n=min(len(datasets["full_spectrum"]), BOT_SAMPLE_CAP), random_state=rng)[feature_cols].values)
        if "gen_bot" in datasets:
            ext_bot_parts.append(datasets["gen_bot"].sample(n=min(len(datasets["gen_bot"]), BOT_SAMPLE_CAP // 2), random_state=rng)[feature_cols].values)
        if "cal_bot" in datasets:
            ext_bot_parts.append(datasets["cal_bot"].sample(n=min(len(datasets["cal_bot"]), BOT_SAMPLE_CAP // 2), random_state=rng)[feature_cols].values)
        if "acpc_bot" in datasets:
            ext_bot_parts.append(datasets["acpc_bot"].sample(n=min(len(datasets["acpc_bot"]), BOT_SAMPLE_CAP), random_state=rng)[feature_cols].values)

        ext_human = np.vstack(ext_human_parts) if ext_human_parts else np.empty((0, len(feature_cols)))
        ext_bot = np.vstack(ext_bot_parts) if ext_bot_parts else np.empty((0, len(feature_cols)))

        cv_results = evaluate_gold_loocv(
            datasets["gold"], ext_human, ext_bot, feature_cols, transform_meta, rf_kwargs
        )
        for r in cv_results:
            marker = " <-- diff bot" if "05-08" in r["date"] else ""
            print(f"  {r['date']}: AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  BotDetect={r['bot_detect_rate']:.3f}{marker}")
        mean_auc = np.mean([r["auc"] for r in cv_results if not np.isnan(r["auc"])])
        min_auc = np.min([r["auc"] for r in cv_results if not np.isnan(r["auc"])])
        print(f"  Mean AUC: {mean_auc:.4f} | Min AUC: {min_auc:.4f}")

    # Cross-domain generalization (training data sanity check)
    print("\nCross-domain generalization (train data):")
    if "zenodo" in datasets:
        zen_t = apply_transform(datasets["zenodo"][feature_cols].values, feature_cols, transform_meta)
        zen_human = (rf.predict_proba(zen_t)[:, 1] < 0.5).mean()
        print(f"  Zenodo train → human: {zen_human*100:.1f}%")
    if "public" in datasets:
        pub_t = apply_transform(datasets["public"][feature_cols].values, feature_cols, transform_meta)
        pub_human = (rf.predict_proba(pub_t)[:, 1] < 0.5).mean()
        print(f"  Public train → human: {pub_human*100:.1f}%")

    # ── Unseen test evaluation ──
    test_results = {}
    test_files = {
        "zenodo_test": ("zenodo_test_features.parquet", 0),
        "public_test": ("public_test_features.parquet", 0),
        "acpc_bot_test": ("acpc_bot_test_features.parquet", 1),
        "may8_gold_test": ("may8_gold_test_features.parquet", None),
    }
    any_test = False
    for tname, (tfile, true_label) in test_files.items():
        tpath = TEST_DIR / tfile
        if not tpath.is_file():
            continue
        any_test = True
        tdf = pd.read_parquet(tpath)
        missing = [c for c in feature_cols if c not in tdf.columns]
        if missing:
            print(f"  SKIP {tname}: missing features")
            continue
        tX = apply_transform(tdf[feature_cols].values, feature_cols, transform_meta)
        tproba = rf.predict_proba(tX)[:, 1]
        if tname == "may8_gold_test":
            labels = tdf["label"].values
            may8_h = tproba[labels == 0]
            may8_b = tproba[labels == 1]
            test_results[tname] = {
                "n": len(tdf),
                "human_fpr_pct": round(float((may8_h >= 0.5).mean()) * 100, 3) if len(may8_h) else None,
                "bot_recall_pct": round(float((may8_b >= 0.5).mean()) * 100, 2) if len(may8_b) else None,
                "bot_mean_score": round(float(may8_b.mean()), 4) if len(may8_b) else None,
                "human_mean_score": round(float(may8_h.mean()), 4) if len(may8_h) else None,
            }
        elif true_label == 0:
            correct = float((tproba < 0.5).mean())
            fpr = 1.0 - correct
            test_results[tname] = {"correct_pct": round(correct * 100, 2), "fpr_pct": round(fpr * 100, 3), "n": len(tdf)}
        else:
            recall = float((tproba >= 0.5).mean())
            test_results[tname] = {"recall_pct": round(recall * 100, 2), "n": len(tdf), "mean_score": round(float(tproba.mean()), 4)}

    if any_test:
        print("\n" + "=" * 70)
        print("UNSEEN TEST SET EVALUATION")
        print("=" * 70)
        for tname, tres in test_results.items():
            if tname == "may8_gold_test":
                print(
                    f"  {tname:20s}: {tres['n']:5d} chunks | "
                    f"bot recall={tres['bot_recall_pct']:.1f}% | "
                    f"human FPR={tres['human_fpr_pct']:.3f}%"
                )
            elif "fpr_pct" in tres:
                print(f"  {tname:20s}: {tres['n']:5d} chunks | human correct={tres['correct_pct']:.1f}% | FPR={tres['fpr_pct']:.3f}%")
            else:
                print(f"  {tname:20s}: {tres['n']:5d} chunks | bot recall={tres['recall_pct']:.1f}% | mean_score={tres['mean_score']:.4f}")

    may8_holdout = load_may8_gold_test()
    if may8_holdout is not None and len(may8_holdout) and "may8_gold_test" not in test_results:
        missing = [c for c in feature_cols if c not in may8_holdout.columns]
        if not missing:
            may8_X = apply_transform(may8_holdout[feature_cols].values, feature_cols, transform_meta)
            may8_proba = rf.predict_proba(may8_X)[:, 1]
            may8_h = may8_proba[may8_holdout["label"].values == 0]
            may8_b = may8_proba[may8_holdout["label"].values == 1]
            print(f"\n  May-8 gold hold-out (dataset/test):")
            if len(may8_h) > 0:
                print(f"    Human scores: min={may8_h.min():.4f} max={may8_h.max():.4f} mean={may8_h.mean():.4f}")
                print(f"    Human FPR @0.5: {(may8_h >= 0.5).mean()*100:.1f}%")
            if len(may8_b) > 0:
                print(f"    Bot   scores: min={may8_b.min():.4f} max={may8_b.max():.4f} mean={may8_b.mean():.4f}")
                print(f"    Bot recall @0.5:  {(may8_b >= 0.5).mean()*100:.1f}%")

    # Save artifacts
    print(f"\nSaving model bundle to {output_dir}...")
    import joblib
    joblib.dump(rf, output_dir / "model.joblib")
    print(f"  Saved model.joblib")

    (output_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    print(f"  Saved feature_cols.json ({len(feature_cols)} features)")

    (output_dir / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )
    print(f"  Saved transform_meta.json")

    summary = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_features": len(feature_cols),
        "n_training_samples": int(len(X_raw)),
        "data_sources": data_meta["sources"],
        "train_auc": round(float(train_auc), 6),
        "train_accuracy": round(float(train_acc), 6),
        "selected_threshold": optimal_threshold,
        "cv_results": cv_results if "gold" in datasets else [],
        "cv_mean_auc": round(float(mean_auc), 6) if "gold" in datasets else None,
        "cv_min_auc": round(float(min_auc), 6) if "gold" in datasets else None,
        "unseen_test_results": test_results if any_test else {},
        "model_type": "RandomForestClassifier",
        "model_params": rf_params_for_summary(rf_kwargs),
    }
    (output_dir / "retrain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  Saved retrain_summary.json")

    # Feature importances
    print("\nTop 15 feature importances:")
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx[:15], 1):
        print(f"  {rank:2d}. {feature_cols[i]:<35s} {importances[i]:.4f}")

    print("\n" + "=" * 70)
    print("DEPLOYMENT INSTRUCTIONS:")
    print("=" * 70)
    print(f"  1. Set POKER44_MINER_MODEL_PATH={output_dir / 'model.joblib'}")
    print(f"  2. transform_meta.json will be auto-loaded from same directory")
    print(f"  3. Set POKER44_MINER_OTHER_ONLY=0 to use hybrid model")
    print(f"  4. Set POKER44_MINER_REQUIRE_MODEL=1 to enforce model loading")


if __name__ == "__main__":
    main()
