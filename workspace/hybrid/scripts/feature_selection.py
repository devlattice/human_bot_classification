"""Multi-feature selection pipeline for Hybrid Miner B.

Steps:
  2A: Pre-filter (remove banned, constant, highly correlated)
  2B: Sequential Forward Selection with leave-one-day-out CV
  2C: Robustness filter (direction consistency with generated bots)

Usage:
    python workspace/hybrid/scripts/feature_selection.py

Inputs:
    workspace/hybrid/dataset/train/gold_features.parquet   (Apr30–May7)
    workspace/hybrid/dataset/train/zenodo_features.parquet
    workspace/hybrid/dataset/train/generated_bot_features.parquet

Hold-out (not used in this script yet; for v3 selection):
    workspace/hybrid/dataset/test/may8_gold_test_features.parquet

Outputs:
    workspace/hybrid/selected_features.json
    workspace/hybrid/feature_selection_report.txt
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "gold_features.parquet"
ZENODO_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "zenodo_features.parquet"
GEN_BOT_PATH = REPO_ROOT / "workspace" / "hybrid" / "generated_bot_features.parquet"
OUTPUT_FEATURES = REPO_ROOT / "workspace" / "hybrid" / "selected_features.json"
OUTPUT_REPORT = REPO_ROOT / "workspace" / "hybrid" / "feature_selection_report.txt"

BANNED_PREFIXES = ("other_ratio", "n_actions")
CORR_THRESHOLD = 0.95
MAX_FEATURES = 30
MIN_FEATURES = 15  # always select at least this many
MIN_IMPROVEMENT = 0.0005  # stop AFTER min_features if gain < this


def load_data():
    gold = pd.read_parquet(GOLD_PATH)
    zenodo = pd.read_parquet(ZENODO_PATH)
    gen_bots = pd.read_parquet(GEN_BOT_PATH)
    return gold, zenodo, gen_bots


# ─────────────────────────────────────────────────────────────────────
# STEP 2A: Pre-filter
# ─────────────────────────────────────────────────────────────────────

def prefilter_features(gold: pd.DataFrame) -> list[str]:
    """Remove banned, constant, and highly correlated features."""
    meta_cols = {"label", "source", "date"}
    all_feats = [c for c in gold.columns if c not in meta_cols]

    # Remove banned
    candidates = [
        f for f in all_feats
        if not any(f.startswith(p) for p in BANNED_PREFIXES)
    ]
    print(f"  After removing banned prefixes: {len(candidates)}")

    # Remove constant (std ~ 0)
    feat_data = gold[candidates]
    stds = feat_data.std()
    constant = stds[stds < 1e-8].index.tolist()
    candidates = [f for f in candidates if f not in constant]
    print(f"  After removing constant features: {len(candidates)} (dropped {len(constant)})")

    # Remove highly correlated (keep first of each pair)
    corr_matrix = feat_data[candidates].corr().abs()
    to_drop = set()
    for i in range(len(candidates)):
        if candidates[i] in to_drop:
            continue
        for j in range(i + 1, len(candidates)):
            if candidates[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
                to_drop.add(candidates[j])

    candidates = [f for f in candidates if f not in to_drop]
    print(f"  After removing correlated (|r|>{CORR_THRESHOLD}): {len(candidates)} (dropped {len(to_drop)})")

    return candidates


# ─────────────────────────────────────────────────────────────────────
# STEP 2B: Sequential Forward Selection with Leave-One-Day-Out CV
# ─────────────────────────────────────────────────────────────────────

def leave_one_day_out_auc(
    gold: pd.DataFrame,
    features: list[str],
) -> tuple[float, list[float]]:
    """Train RF on 8 days, test on 1. Return mean AUC and per-day AUCs."""
    dates = sorted(gold["date"].unique())
    day_aucs = []

    for test_date in dates:
        train = gold[gold["date"] != test_date]
        test = gold[gold["date"] == test_date]

        X_train = train[features].values
        y_train = train["label"].values
        X_test = test[features].values
        y_test = test["label"].values

        if len(np.unique(y_test)) < 2:
            continue

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        day_aucs.append(auc)

    mean_auc = float(np.mean(day_aucs)) if day_aucs else 0.0
    return mean_auc, day_aucs


def sequential_forward_selection(
    gold: pd.DataFrame,
    candidates: list[str],
) -> list[tuple[str, float, float]]:
    """Greedy forward selection. Returns [(feature, auc_after_adding, improvement), ...]"""
    selected = []
    selected_names = []
    remaining = list(candidates)
    best_auc = 0.5

    print(f"\n  Starting forward selection with {len(remaining)} candidates...")
    print(f"  {'Step':<5} {'Feature added':<35} {'AUC':>7} {'Gain':>7} {'Min-day':>8}")
    print(f"  {'-'*70}")

    for step in range(MAX_FEATURES):
        best_feat = None
        best_new_auc = best_auc
        best_day_aucs = []

        for feat in remaining:
            trial_feats = selected_names + [feat]
            auc, day_aucs = leave_one_day_out_auc(gold, trial_feats)

            if auc > best_new_auc:
                best_new_auc = auc
                best_feat = feat
                best_day_aucs = day_aucs

        if best_feat is None:
            print(f"  Stopping: no candidate improves AUC")
            break

        if step >= MIN_FEATURES and (best_new_auc - best_auc) < MIN_IMPROVEMENT:
            print(f"  Stopping at step {step+1}: gain < {MIN_IMPROVEMENT} after {MIN_FEATURES} features")
            break

        improvement = best_new_auc - best_auc
        min_day = min(best_day_aucs) if best_day_aucs else 0.0
        selected.append((best_feat, best_new_auc, improvement))
        selected_names.append(best_feat)
        remaining.remove(best_feat)
        best_auc = best_new_auc

        print(f"  {step+1:<5} {best_feat:<35} {best_new_auc:>7.4f} +{improvement:>6.4f} {min_day:>8.4f}")

    return selected


# ─────────────────────────────────────────────────────────────────────
# STEP 2C: Robustness Filter (direction consistency)
# ─────────────────────────────────────────────────────────────────────

def robustness_filter(
    gold: pd.DataFrame,
    zenodo: pd.DataFrame,
    gen_bots: pd.DataFrame,
    selected_features: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Check if feature separates humans from bots in the same direction
    across gold domain AND generated domain.

    Returns: (robust, weak, rejected)
      - robust: same direction, strong signal in both domains
      - weak: direction flips but effect is small (keep with caution)
      - rejected: strong opposite direction (dangerous)
    """
    gold_humans = gold[gold["label"] == 0]
    gold_bots = gold[gold["label"] == 1]

    robust = []
    weak = []
    rejected = []

    for feat in selected_features:
        gold_h_mean = gold_humans[feat].mean()
        gold_b_mean = gold_bots[feat].mean()
        gold_diff = gold_b_mean - gold_h_mean
        gold_pooled_std = gold[feat].std()
        gold_effect = abs(gold_diff) / (gold_pooled_std + 1e-8)

        if feat not in zenodo.columns or feat not in gen_bots.columns:
            robust.append(feat)
            continue

        zen_h_mean = zenodo[feat].mean()
        gen_b_mean = gen_bots[feat].mean()
        gen_diff = gen_b_mean - zen_h_mean

        same_direction = (gold_diff * gen_diff) >= 0

        if same_direction:
            robust.append(feat)
        else:
            # Direction flips — but how strong is the flip?
            gen_pooled_std = pd.concat([zenodo[[feat]], gen_bots[[feat]]])[feat].std()
            gen_effect = abs(gen_diff) / (gen_pooled_std + 1e-8)

            if gen_effect < 0.3:
                # Weak flip (small effect in generated domain) → keep with caution
                weak.append(feat)
            else:
                rejected.append(feat)

    return robust, weak, rejected


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log("=" * 70)
    log("FEATURE SELECTION PIPELINE — Hybrid Miner B")
    log("=" * 70)
    log("")

    # Load data
    gold, zenodo, gen_bots = load_data()
    log(f"Gold: {len(gold)} chunks ({(gold['label']==0).sum()} human, {(gold['label']==1).sum()} bot)")
    log(f"Zenodo: {len(zenodo)} human chunks")
    log(f"Generated bots: {len(gen_bots)} chunks")
    log(f"Days in gold: {sorted(gold['date'].unique())}")
    log("")

    # Step 2A
    log("─" * 70)
    log("STEP 2A: Pre-filter")
    log("─" * 70)
    candidates = prefilter_features(gold)
    log(f"  Final candidates: {len(candidates)}")
    log("")

    # Step 2B
    log("─" * 70)
    log("STEP 2B: Sequential Forward Selection (leave-one-day-out CV)")
    log("─" * 70)
    selection_results = sequential_forward_selection(gold, candidates)
    selected_names = [name for name, _, _ in selection_results]
    log(f"\n  Selected {len(selected_names)} features")
    log("")

    # Final AUC with all selected features
    final_auc, final_day_aucs = leave_one_day_out_auc(gold, selected_names)
    dates = sorted(gold["date"].unique())
    log(f"  Final leave-one-day-out AUC: {final_auc:.4f}")
    log(f"  Per-day AUCs:")
    for d, a in zip(dates, final_day_aucs):
        marker = " ← WORST" if a == min(final_day_aucs) else ""
        log(f"    {d}: {a:.4f}{marker}")
    log("")

    # Step 2C
    log("─" * 70)
    log("STEP 2C: Robustness Filter (direction consistency)")
    log("─" * 70)
    robust_features, weak_features, rejected_features = robustness_filter(
        gold, zenodo, gen_bots, selected_names
    )
    log(f"  Robust (same direction): {len(robust_features)}")
    log(f"  Weak (small flip, kept with caution): {len(weak_features)}")
    log(f"  Rejected (strong opposite direction): {len(rejected_features)}")
    if weak_features:
        log(f"  Weak features: {weak_features}")
    if rejected_features:
        log(f"  Rejected features: {rejected_features}")
    log("")

    # Final feature set: robust + weak (weak are kept for now, can be dropped later)
    final_features = robust_features + weak_features

    # Final validation
    if final_features:
        robust_auc, robust_day_aucs = leave_one_day_out_auc(gold, final_features)
        log(f"  AUC with robust+weak features: {robust_auc:.4f}")
        log(f"  AUC with ALL selected (before filter): {final_auc:.4f}")
    else:
        log("  WARNING: No features survived robustness filter!")
        log("  Falling back to all selected features (profile-dependent risk accepted)")
        final_features = selected_names
        robust_auc = final_auc
        robust_day_aucs = final_day_aucs

    log("")
    log("─" * 70)
    log("FINAL RESULT")
    log("─" * 70)
    log(f"  Final features ({len(final_features)}):")
    for i, f in enumerate(final_features):
        marker = " [weak]" if f in weak_features else ""
        log(f"    {i+1:2d}. {f}{marker}")
    log(f"\n  Leave-one-day-out AUC: {robust_auc:.4f}")
    log(f"  Min single-day AUC: {min(robust_day_aucs) if robust_day_aucs else 0:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.1f}s")

    # Save outputs
    output = {
        "selected_features": final_features,
        "robust_features": robust_features,
        "weak_features": weak_features,
        "rejected_features": rejected_features,
        "auc": robust_auc,
        "per_day_auc": dict(zip(dates, [float(a) for a in robust_day_aucs])) if robust_day_aucs else {},
        "selection_history": [
            {"feature": name, "auc": float(auc), "gain": float(gain)}
            for name, auc, gain in selection_results
        ],
        "prefilter_candidates": len(candidates),
    }

    OUTPUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURES, "w") as f:
        json.dump(output, f, indent=2)

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(report_lines))

    log(f"\n  Saved: {OUTPUT_FEATURES}")
    log(f"  Saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
