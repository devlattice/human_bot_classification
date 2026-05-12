"""Feature selection v2: Two approaches to find robust features.

Option A: Exclude obvious profile-dependent features (fold_ratio_*) and re-select
Option B: Maximize MIN-day AUC (forces features that work on ALL bot profiles)

Also includes human generalization check (gold humans vs zenodo humans).

Usage:
    python workspace/hybrid/scripts/feature_selection_v2.py

Outputs:
    workspace/hybrid/selected_features_v2.json
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
PUBLIC_PATH = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "public_features.parquet"
GEN_BOT_PATH = REPO_ROOT / "workspace" / "hybrid" / "generated_bot_features.parquet"
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "selected_features_v2.json"

BANNED_PREFIXES = ("other_ratio", "n_actions")
PROFILE_DEPENDENT_PATTERNS = ("fold_ratio",)  # Option A excludes these
CORR_THRESHOLD = 0.95
MAX_FEATURES = 25
MIN_FEATURES = 10
MIN_IMPROVEMENT = 0.0003


def load_data():
    gold = pd.read_parquet(GOLD_PATH)
    zenodo = pd.read_parquet(ZENODO_PATH)
    public = pd.read_parquet(PUBLIC_PATH)
    gen_bots = pd.read_parquet(GEN_BOT_PATH)
    return gold, zenodo, public, gen_bots


def prefilter(gold: pd.DataFrame, exclude_patterns: tuple = ()) -> list[str]:
    meta_cols = {"label", "source", "date"}
    all_feats = [c for c in gold.columns if c not in meta_cols]

    # Remove banned + optional profile-dependent
    banned = BANNED_PREFIXES + exclude_patterns
    candidates = [f for f in all_feats if not any(f.startswith(p) for p in banned)]

    # Remove constant
    stds = gold[candidates].std()
    constant = stds[stds < 1e-8].index.tolist()
    candidates = [f for f in candidates if f not in constant]

    # Remove highly correlated
    corr_matrix = gold[candidates].corr().abs()
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
    return candidates


def leave_one_day_out_auc(gold: pd.DataFrame, features: list[str]) -> tuple[float, list[float], list[str]]:
    """Returns (mean_auc, per_day_aucs, dates)."""
    dates = sorted(gold["date"].unique())
    day_aucs = []

    for test_date in dates:
        train = gold[gold["date"] != test_date]
        test = gold[gold["date"] == test_date]
        X_train, y_train = train[features].values, train["label"].values
        X_test, y_test = test[features].values, test["label"].values

        if len(np.unique(y_test)) < 2:
            day_aucs.append(0.5)
            continue

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        day_aucs.append(roc_auc_score(y_test, proba))

    return float(np.mean(day_aucs)), day_aucs, dates


def forward_selection_mean_auc(gold, candidates):
    """Standard: maximize MEAN AUC across days."""
    selected = []
    remaining = list(candidates)
    best_auc = 0.5

    for step in range(MAX_FEATURES):
        best_feat = None
        best_new_auc = best_auc

        for feat in remaining:
            trial = selected + [feat]
            auc, _, _ = leave_one_day_out_auc(gold, trial)
            if auc > best_new_auc:
                best_new_auc = auc
                best_feat = feat

        if best_feat is None:
            break
        if step >= MIN_FEATURES and (best_new_auc - best_auc) < MIN_IMPROVEMENT:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_auc = best_new_auc
        _, day_aucs, _ = leave_one_day_out_auc(gold, selected)
        min_day = min(day_aucs)
        print(f"    {step+1:<3} {best_feat:<35} mean={best_new_auc:.4f} min={min_day:.4f}")

    return selected


def forward_selection_min_auc(gold, candidates):
    """Option B: maximize MIN single-day AUC (robust to rotation)."""
    selected = []
    remaining = list(candidates)
    best_min_auc = 0.0

    for step in range(MAX_FEATURES):
        best_feat = None
        best_new_min = best_min_auc
        best_new_mean = 0.0

        for feat in remaining:
            trial = selected + [feat]
            mean_auc, day_aucs, _ = leave_one_day_out_auc(gold, trial)
            min_auc = min(day_aucs)
            # Primary: maximize min-day AUC; secondary: maximize mean
            if min_auc > best_new_min or (min_auc == best_new_min and mean_auc > best_new_mean):
                best_new_min = min_auc
                best_new_mean = mean_auc
                best_feat = feat

        if best_feat is None:
            break
        if step >= MIN_FEATURES and (best_new_min - best_min_auc) < MIN_IMPROVEMENT:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_min_auc = best_new_min
        print(f"    {step+1:<3} {best_feat:<35} min={best_new_min:.4f} mean={best_new_mean:.4f}")

    return selected


def robustness_check(gold, zenodo, gen_bots, features):
    """Direction consistency check. Returns dict with per-feature results."""
    gold_h = gold[gold["label"] == 0]
    gold_b = gold[gold["label"] == 1]
    results = {}

    for feat in features:
        gold_diff = gold_b[feat].mean() - gold_h[feat].mean()
        if feat in zenodo.columns and feat in gen_bots.columns:
            gen_diff = gen_bots[feat].mean() - zenodo[feat].mean()
            same_dir = (gold_diff * gen_diff) >= 0
            results[feat] = {
                "gold_direction": "bot>human" if gold_diff > 0 else "human>bot",
                "gen_direction": "bot>human" if gen_diff > 0 else "human>bot",
                "consistent": same_dir,
                "gold_effect": abs(gold_diff) / (gold[feat].std() + 1e-8),
                "gen_effect": abs(gen_diff) / (pd.concat([zenodo[[feat]], gen_bots[[feat]]])[feat].std() + 1e-8),
            }
        else:
            results[feat] = {"consistent": True, "note": "not checkable"}

    return results


def human_generalization_check(gold, zenodo, public, features):
    """Check if selected features are stable across different human populations.
    Low KS = feature behaves similarly for all humans (good for generalization).
    """
    from scipy import stats

    gold_h = gold[gold["label"] == 0]
    results = {}

    for feat in features:
        if feat not in zenodo.columns:
            results[feat] = {"ks_gold_zenodo": None, "ks_gold_public": None}
            continue

        g = gold_h[feat].dropna().values
        z = zenodo[feat].dropna().values
        p = public[feat].dropna().values if feat in public.columns else np.array([])

        ks_gz, _ = stats.ks_2samp(g, z) if len(z) > 10 else (None, None)
        ks_gp, _ = stats.ks_2samp(g, p) if len(p) > 10 else (None, None)

        results[feat] = {
            "ks_gold_zenodo": float(ks_gz) if ks_gz is not None else None,
            "ks_gold_public": float(ks_gp) if ks_gp is not None else None,
            "gold_h_mean": float(g.mean()),
            "zenodo_mean": float(z.mean()),
            "human_stable": ks_gz is not None and ks_gz < 0.5,
        }

    return results


def main():
    t0 = time.time()
    gold, zenodo, public, gen_bots = load_data()

    print("=" * 70)
    print("FEATURE SELECTION V2 — Two approaches + human generalization")
    print("=" * 70)
    print(f"Gold: {len(gold)} | Zenodo: {len(zenodo)} | Public: {len(public)} | Gen bots: {len(gen_bots)}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # OPTION A: Exclude fold_ratio_* entirely
    # ═══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("OPTION A: Exclude fold_ratio_* (profile-dependent) + forward selection")
    print("═" * 70)
    candidates_a = prefilter(gold, exclude_patterns=PROFILE_DEPENDENT_PATTERNS)
    print(f"  Candidates after excluding fold_ratio_*: {len(candidates_a)}")
    print()
    selected_a = forward_selection_mean_auc(gold, candidates_a)
    auc_a, days_a, dates = leave_one_day_out_auc(gold, selected_a)
    print(f"\n  Selected {len(selected_a)} features, AUC={auc_a:.4f}, min-day={min(days_a):.4f}")
    print(f"  Per-day: {dict(zip(dates, [f'{a:.4f}' for a in days_a]))}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # OPTION B: Maximize MIN-day AUC (all features allowed)
    # ═══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("OPTION B: Maximize MIN-day AUC (forces robustness to rotation)")
    print("═" * 70)
    candidates_b = prefilter(gold, exclude_patterns=())  # allow all
    print(f"  Candidates: {len(candidates_b)}")
    print()
    selected_b = forward_selection_min_auc(gold, candidates_b)
    auc_b, days_b, _ = leave_one_day_out_auc(gold, selected_b)
    print(f"\n  Selected {len(selected_b)} features, AUC={auc_b:.4f}, min-day={min(days_b):.4f}")
    print(f"  Per-day: {dict(zip(dates, [f'{a:.4f}' for a in days_b]))}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # ROBUSTNESS CHECK for both
    # ═══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("ROBUSTNESS CHECK (direction consistency with generated bots)")
    print("═" * 70)

    print("\n  Option A features:")
    rob_a = robustness_check(gold, zenodo, gen_bots, selected_a)
    for feat, info in rob_a.items():
        status = "✓" if info.get("consistent") else "✗"
        print(f"    {status} {feat:<35} gold={info.get('gold_direction','')} gen={info.get('gen_direction','')}")

    print("\n  Option B features:")
    rob_b = robustness_check(gold, zenodo, gen_bots, selected_b)
    for feat, info in rob_b.items():
        status = "✓" if info.get("consistent") else "✗"
        print(f"    {status} {feat:<35} gold={info.get('gold_direction','')} gen={info.get('gen_direction','')}")

    # ═══════════════════════════════════════════════════════════════════
    # HUMAN GENERALIZATION CHECK
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 70)
    print("HUMAN GENERALIZATION (gold humans vs zenodo/public humans)")
    print("═" * 70)
    all_selected = list(set(selected_a + selected_b))

    print(f"\n  {'Feature':<35} {'KS(gold↔zenodo)':>15} {'KS(gold↔public)':>15} {'Stable?':<8}")
    print(f"  {'-'*78}")
    human_gen = human_generalization_check(gold, zenodo, public, all_selected)
    for feat in sorted(all_selected):
        info = human_gen[feat]
        ks_z = f"{info['ks_gold_zenodo']:.3f}" if info['ks_gold_zenodo'] is not None else "N/A"
        ks_p = f"{info['ks_gold_public']:.3f}" if info['ks_gold_public'] is not None else "N/A"
        stable = "YES" if info.get("human_stable") else "NO"
        print(f"  {feat:<35} {ks_z:>15} {ks_p:>15} {stable:<8}")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("═" * 70)
    print("SUMMARY COMPARISON")
    print("═" * 70)
    robust_a = [f for f in selected_a if rob_a[f].get("consistent")]
    robust_b = [f for f in selected_b if rob_b[f].get("consistent")]
    human_stable_a = [f for f in selected_a if human_gen.get(f, {}).get("human_stable")]
    human_stable_b = [f for f in selected_b if human_gen.get(f, {}).get("human_stable")]

    print(f"\n  {'Metric':<40} {'Option A':>12} {'Option B':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Features selected':<40} {len(selected_a):>12} {len(selected_b):>12}")
    print(f"  {'Mean AUC (leave-one-day-out)':<40} {auc_a:>12.4f} {auc_b:>12.4f}")
    print(f"  {'Min single-day AUC':<40} {min(days_a):>12.4f} {min(days_b):>12.4f}")
    print(f"  {'Direction-robust features':<40} {len(robust_a):>12} {len(robust_b):>12}")
    print(f"  {'Human-stable features (KS<0.5)':<40} {len(human_stable_a):>12} {len(human_stable_b):>12}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save
    output = {
        "option_a": {
            "description": "Exclude fold_ratio_*, maximize mean AUC",
            "features": selected_a,
            "auc_mean": auc_a,
            "auc_min": min(days_a),
            "per_day_auc": dict(zip(dates, [float(a) for a in days_a])),
            "robustness": {f: rob_a[f] for f in selected_a},
            "robust_count": len(robust_a),
            "human_stable_count": len(human_stable_a),
        },
        "option_b": {
            "description": "All features, maximize MIN-day AUC",
            "features": selected_b,
            "auc_mean": auc_b,
            "auc_min": min(days_b),
            "per_day_auc": dict(zip(dates, [float(a) for a in days_b])),
            "robustness": {f: rob_b[f] for f in selected_b},
            "robust_count": len(robust_b),
            "human_stable_count": len(human_stable_b),
        },
        "human_generalization": human_gen,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
