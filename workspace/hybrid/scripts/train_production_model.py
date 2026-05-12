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

GOLD_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "gold_features.parquet"
ZENODO_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "zenodo_features.parquet"
PUBLIC_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "public_features.parquet"
FULL_SPECTRUM_PATH = REPO_ROOT / "workspace" / "hybrid" / "full_spectrum_bot_features.parquet"
GEN_BOT_PATH = REPO_ROOT / "workspace" / "hybrid" / "generated_bot_features.parquet"
CAL_BOT_PATH = REPO_ROOT / "workspace" / "hybrid" / "calibrated_bot_features.parquet"

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

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
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

    print("\nTraining RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=args.seed, n_jobs=-1, class_weight="balanced",
    )
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

        ext_human = np.vstack(ext_human_parts) if ext_human_parts else np.empty((0, len(feature_cols)))
        ext_bot = np.vstack(ext_bot_parts) if ext_bot_parts else np.empty((0, len(feature_cols)))

        cv_results = evaluate_gold_loocv(datasets["gold"], ext_human, ext_bot, feature_cols, transform_meta)
        for r in cv_results:
            marker = " <-- diff bot" if "05-08" in r["date"] else ""
            print(f"  {r['date']}: AUC={r['auc']:.4f}  Acc={r['accuracy']:.4f}  BotDetect={r['bot_detect_rate']:.3f}{marker}")
        mean_auc = np.mean([r["auc"] for r in cv_results if not np.isnan(r["auc"])])
        min_auc = np.min([r["auc"] for r in cv_results if not np.isnan(r["auc"])])
        print(f"  Mean AUC: {mean_auc:.4f} | Min AUC: {min_auc:.4f}")

    # Cross-domain generalization
    print("\nCross-domain generalization:")
    if "zenodo" in datasets:
        zen_t = apply_transform(datasets["zenodo"][feature_cols].values, feature_cols, transform_meta)
        zen_human = (rf.predict_proba(zen_t)[:, 1] < 0.5).mean()
        print(f"  Zenodo → human: {zen_human*100:.1f}%")
    if "public" in datasets:
        pub_t = apply_transform(datasets["public"][feature_cols].values, feature_cols, transform_meta)
        pub_human = (rf.predict_proba(pub_t)[:, 1] < 0.5).mean()
        print(f"  Public → human: {pub_human*100:.1f}%")

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
        "model_type": "RandomForestClassifier",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 15,
            "class_weight": "balanced",
        },
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
