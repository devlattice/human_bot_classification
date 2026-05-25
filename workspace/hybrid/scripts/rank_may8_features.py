"""Rank chunk features for May-8 hard-bot separability (one JSON report).

Scores May-8 with a hold-out RF (May-8 excluded from fit), splits gold bots into
hard/easy at --score-threshold, then ranks every numeric feature by:

  - KS / AUC: hard_bot vs may8_human  (primary objective)
  - KS / AUC: easy_bot vs may8_human
  - KS / AUC: all gold human vs bot
  - |corr|(feature, holdout bot score) on May-8 bots

Output:
  workspace/hybrid/bot_system/data/may8_feature_ranking.json

Usage:
  python workspace/hybrid/scripts/rank_may8_features.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))

import train_production_model as tpm  # noqa: E402

DEFAULT_OUT = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_feature_ranking.json"
META_COLS = frozenset({"label", "source", "date", "chunk_idx"})


def numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def symmetric_auc(y: np.ndarray, x: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or float(np.std(x)) < 1e-12:
        return 0.5
    auc = float(roc_auc_score(y, x))
    return max(auc, 1.0 - auc)


def fit_holdout_rf(
    gold: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[RandomForestClassifier, dict]:
    """Train on gold \\ May-8 + zenodo subsample; same recipe as 09_train_and_check."""
    train_gold = gold[~gold["date"].astype(str).str.contains("2026-05-08")]
    zen_path = tpm.ZENODO_PATH
    parts_h = [train_gold[train_gold["label"] == 0][feature_cols]]
    parts_b = [train_gold[train_gold["label"] == 1][feature_cols]]
    if zen_path.is_file():
        zen = pd.read_parquet(zen_path)
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=seed)[feature_cols])

    Xh = pd.concat(parts_h, ignore_index=True)
    Xb = pd.concat(parts_b, ignore_index=True)
    X_raw = pd.concat([Xh, Xb], ignore_index=True).values
    y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_t, y)
    return rf, transform_meta


def rank_features(
    gold: pd.DataFrame,
    may8: pd.DataFrame,
    hard: pd.DataFrame,
    easy: pd.DataFrame,
    may8_h: pd.DataFrame,
    bots: pd.DataFrame,
    feature_cols: list[str],
) -> list[dict]:
    gold_h = gold[gold["label"] == 0]
    gold_b = gold[gold["label"] == 1]
    rows: list[dict] = []

    for c in feature_cols:
        if c not in may8.columns:
            continue
        h = hard[c].fillna(0.0).values.astype(float)
        e = easy[c].fillna(0.0).values.astype(float)
        u = may8_h[c].fillna(0.0).values.astype(float)
        gh = gold_h[c].fillna(0.0).values.astype(float)
        gb = gold_b[c].fillna(0.0).values.astype(float)

        y_hu = np.concatenate([np.zeros(len(u)), np.ones(len(h))])
        x_hu = np.concatenate([u, h])
        y_eu = np.concatenate([np.zeros(len(u)), np.ones(len(e))]) if len(e) else np.array([])
        x_eu = np.concatenate([u, e]) if len(e) else np.array([])
        y_gb = np.concatenate([np.zeros(len(gh)), np.ones(len(gb))])

        auc_hard = symmetric_auc(y_hu, x_hu)
        auc_easy = symmetric_auc(y_eu, x_eu) if len(e) >= 10 else None
        auc_gold = symmetric_auc(y_gb, np.concatenate([gh, gb]))

        ks_hard = float(ks_2samp(h, u).statistic) if len(h) and len(u) else 0.0
        ks_easy = float(ks_2samp(e, u).statistic) if len(e) >= 10 and len(u) else None
        ks_gold = float(ks_2samp(gb, gh).statistic)

        corr = 0.0
        if len(bots) > 5:
            bv = bots[c].fillna(0.0).values.astype(float)
            bs = bots["_holdout_score"].values.astype(float)
            if float(np.std(bv)) > 1e-12:
                corr = float(abs(np.corrcoef(bv, bs)[0, 1]))

        # Primary score: separate hard from human without confusing easy bots
        hard_sep = auc_hard
        easy_confuse = (auc_easy - 0.5) if auc_easy is not None else 0.0
        composite = hard_sep - 0.35 * max(0.0, easy_confuse) + 0.1 * corr

        rows.append({
            "feature": c,
            "in_robust": c in tpm.ROBUST_FEATURES,
            "auc_hard_vs_may8_human": round(auc_hard, 4),
            "auc_easy_vs_may8_human": round(auc_easy, 4) if auc_easy is not None else None,
            "auc_gold_human_vs_bot": round(auc_gold, 4),
            "ks_hard_vs_may8_human": round(ks_hard, 4),
            "ks_easy_vs_may8_human": round(ks_easy, 4) if ks_easy is not None else None,
            "ks_gold_human_vs_bot": round(ks_gold, 4),
            "corr_abs_bot_score": round(corr, 4),
            "composite_rank_score": round(composite, 4),
            "hard_mean": round(float(h.mean()), 6),
            "human_mean": round(float(u.mean()), 6),
            "easy_mean": round(float(e.mean()), 6) if len(e) else None,
        })

    rows.sort(key=lambda r: -r["composite_rank_score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-set",
        choices=("robust", "all_numeric"),
        default="all_numeric",
        help="robust: tpm.ROBUST_FEATURES; all_numeric: every numeric col in gold parquet.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gold_path = tpm.GOLD_PATH
    if not gold_path.is_file():
        print(f"[error] missing {gold_path}")
        return 1

    gold = pd.read_parquet(gold_path)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")].copy()

    if args.feature_set == "robust":
        candidates = list(tpm.ROBUST_FEATURES)
    else:
        candidates = numeric_features(gold)

    avail = set(candidates)
    for sub in (gold, may8):
        avail &= set(sub.columns)
    feature_cols = [c for c in candidates if c in avail]

    print(f"[load] gold={len(gold)}  may8={len(may8)}  features={len(feature_cols)}")
    rf, tm = fit_holdout_rf(gold, feature_cols, args.seed)

    Xt = tpm.apply_transform(may8[feature_cols].values, feature_cols, tm)
    may8["_holdout_score"] = rf.predict_proba(Xt)[:, 1]

    bots = may8[may8["label"] == 1]
    humans = may8[may8["label"] == 0]
    hard = bots[bots["_holdout_score"] < args.score_threshold]
    easy = bots[bots["_holdout_score"] >= args.score_threshold]
    recall = float((bots["_holdout_score"] >= args.score_threshold).mean()) * 100

    print(f"[holdout] bots={len(bots)} hard={len(hard)} easy={len(easy)} "
          f"recall@{args.score_threshold}={recall:.2f}%")
    print(f"  hard score p50={hard['_holdout_score'].median():.4f}  "
          f"easy p50={easy['_holdout_score'].median():.4f}  "
          f"human max={humans['_holdout_score'].max():.4f}")

    ranked = rank_features(gold, may8, hard, easy, humans, bots, feature_cols)
    robust_in_top20 = sum(1 for r in ranked[:20] if r["in_robust"])

    payload = {
        "version": 1,
        "feature_pipeline": "payload-view-action-leak-tighten-2026-05",
        "score_threshold": args.score_threshold,
        "feature_set": args.feature_set,
        "n_features_ranked": len(ranked),
        "may8_summary": {
            "n_chunks": int(len(may8)),
            "n_bots": int(len(bots)),
            "n_humans": int(len(humans)),
            "n_hard_bots": int(len(hard)),
            "n_easy_bots": int(len(easy)),
            "holdout_recall_pct": round(recall, 3),
            "hard_score_median": round(float(hard["_holdout_score"].median()), 4) if len(hard) else None,
            "easy_score_median": round(float(easy["_holdout_score"].median()), 4) if len(easy) else None,
            "human_score_max": round(float(humans["_holdout_score"].max()), 4),
        },
        "interpretation": {
            "primary_metric": "auc_hard_vs_may8_human (higher = better for catching hard bots)",
            "composite_rank_score": "auc_hard - 0.35*max(0,auc_easy-0.5) + 0.1*|corr|",
            "warning": "High auc_gold but low auc_hard => feature helps average bot, not May-8 hard slice.",
        },
        "robust_features_in_top20": robust_in_top20,
        "top20": ranked[:20],
        "bottom10_for_hard_separation": ranked[-10:],
        "recommended_focus": [
            r["feature"]
            for r in ranked[:25]
            if r["auc_hard_vs_may8_human"] >= 0.55
        ],
        "features": ranked,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("TOP 15 for May-8 HARD bot vs human (hold-out model)")
    print("=" * 72)
    print(f"{'rank':>4}  {'feature':<36}  {'AUC_hard':>8}  {'AUC_gold':>8}  {'robust':>6}")
    for r in ranked[:15]:
        print(
            f"{r['rank']:4d}  {r['feature']:<36}  "
            f"{r['auc_hard_vs_may8_human']:8.4f}  {r['auc_gold_human_vs_bot']:8.4f}  "
            f"{'yes' if r['in_robust'] else 'no':>6}"
        )

    weak_robust = [
        r for r in ranked
        if r["in_robust"] and r["auc_hard_vs_may8_human"] < 0.55
    ]
    print(f"\nROBUST features with weak hard separation (AUC<0.55): {len(weak_robust)}")
    for r in weak_robust[:8]:
        print(f"  {r['feature']}  auc_hard={r['auc_hard_vs_may8_human']}")

    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
