"""Investigate whether the 140 'hard' May-8 bots (score < 0.5) form a single
behavioural cluster or several sub-modes.

- Loads bundle, scores May-8 bots, isolates hard subset.
- Reports score histogram (text bars) for bots and humans.
- Tries KMeans k=1..5 on a behaviour-only feature subset, scoring by
  silhouette + within-cluster sum of squares (elbow).
- Reports cluster centroids (raw mean) on the discriminative features.
- Saves cluster assignments to a CSV for later fingerprinting.

Outputs:
    workspace/hybrid/bot_system/data/may8_hard_clusters.csv
    workspace/hybrid/bot_system/data/may8_hard_clusters_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid" / "scripts"))

import joblib  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

import train_production_model as tpm  # type: ignore  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "round_03_bundle"
DATA_DIR = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data"

# Features that clearly drive the hard/easy split (from 10_diagnose).
BEHAVIORAL_FEATURES = [
    "aggression_factor_mean", "aggression_factor_p90", "aggression_factor_std",
    "aggression_factor_max",
    "bet_ratio_mean", "bet_ratio_p50", "bet_ratio_p90", "bet_ratio_std", "bet_ratio_max",
    "check_ratio_p90", "call_ratio_p90", "fold_ratio_p10",
    "raise_minus_call_mean", "bet_minus_fold_mean",
    "fold_position_mean_mean", "action_entropy_p50", "action_entropy_p90",
    "pot_growth_std", "pot_after_over_stack_mean",
]


def text_hist(values: np.ndarray, bins: int = 20, width: int = 50, lo: float = 0.0, hi: float = 1.0) -> list[str]:
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    mx = max(counts.max(), 1)
    out = []
    for i in range(bins):
        bar = "#" * int(width * counts[i] / mx)
        out.append(f"  [{edges[i]:.2f},{edges[i+1]:.2f})  {counts[i]:4d}  {bar}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--out-csv", type=Path, default=DATA_DIR / "may8_hard_clusters.csv")
    p.add_argument("--out-report", type=Path, default=DATA_DIR / "may8_hard_clusters_report.txt")
    p.add_argument("--max-k", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rf = joblib.load(args.bundle / "lgbm_student.joblib")
    feature_cols = json.loads((args.bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta = json.loads((args.bundle / "transform_meta.json").read_text())

    out_lines: list[str] = []

    def out(msg: str = ""):
        print(msg)
        out_lines.append(msg)

    out(f"[bundle] {args.bundle}")

    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].str.contains("2026-05-08")].copy()
    X = tpm.apply_transform(may8[feature_cols].values, feature_cols, transform_meta)
    may8["proba"] = rf.predict_proba(X)[:, 1]

    bots = may8[may8["label"] == 1].copy()
    humans = may8[may8["label"] == 0].copy()
    hard = bots[bots["proba"] < 0.5].copy()
    easy = bots[bots["proba"] >= 0.5].copy()

    out("\n=== Score histograms on May-8 (bin = 0.05 wide) ===")
    out("\nBot scores  (n={}, mean={:.3f}):".format(len(bots), bots["proba"].mean()))
    for ln in text_hist(bots["proba"].values, bins=20):
        out(ln)
    out("\nHuman scores  (n={}, mean={:.3f}):".format(len(humans), humans["proba"].mean()))
    for ln in text_hist(humans["proba"].values, bins=20):
        out(ln)
    out("\nHard-bot scores (only n={} chunks < 0.5):".format(len(hard)))
    for ln in text_hist(hard["proba"].values, bins=20, lo=0.0, hi=0.5):
        out(ln)

    # ── Cluster the 140 hard bots in behavioural feature space ──────────
    feats = [c for c in BEHAVIORAL_FEATURES if c in hard.columns]
    Xh = hard[feats].fillna(0.0).astype(np.float64).values
    Xs = RobustScaler(quantile_range=(5, 95)).fit_transform(Xh)

    out(f"\n=== KMeans elbow on hard bots  (n={len(hard)}, dim={len(feats)}) ===")
    elbow: dict[int, dict[str, float]] = {}
    for k in range(1, args.max_k + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
        inertia = float(km.inertia_)
        sil = None
        if 2 <= k <= len(Xs) - 1:
            try:
                sil = float(silhouette_score(Xs, km.labels_))
            except Exception:
                sil = None
        elbow[k] = {"inertia": inertia, "silhouette": sil}
        out(f"  k={k}  inertia={inertia:10.2f}  silhouette="
            f"{('%.3f' % sil) if sil is not None else '   n/a'}")

    # Auto-pick best k by silhouette (skip k=1) but cap by a min-cluster-size guard
    def pick_k() -> int:
        cands = [(k, v["silhouette"], v["inertia"]) for k, v in elbow.items()
                 if v["silhouette"] is not None]
        if not cands:
            return 1
        cands.sort(key=lambda r: (-(r[1] or -1), r[2]))
        return cands[0][0]

    best_k = pick_k()
    out(f"\n[auto-picked] k={best_k}")

    km_best = KMeans(n_clusters=best_k, n_init=50, random_state=0).fit(Xs)
    hard["cluster"] = km_best.labels_

    out("\n=== Cluster sizes ===")
    for c, n in sorted(pd.Series(km_best.labels_).value_counts().to_dict().items()):
        out(f"  cluster {c}: {n}")

    # Per-cluster mean of behavioral features + score
    out("\n=== Cluster centroids (raw means) ===")
    header = f"  {'feature':30s}  " + "  ".join(f"c{c:<2d}" + (" "*8)
                                                  for c in sorted(hard['cluster'].unique()))
    out(header)
    for f in feats + ["proba"]:
        if f not in hard.columns:
            continue
        vals = []
        for c in sorted(hard["cluster"].unique()):
            sub = hard[hard["cluster"] == c]
            vals.append(f"{sub[f].mean():10.4f}")
        out(f"  {f:30s}  " + "  ".join(vals))

    # Save assignments
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    hard[["proba", "cluster"] + feats].to_csv(args.out_csv, index=False)
    args.out_report.write_text("\n".join(out_lines))
    out(f"\n[done] csv={args.out_csv}\n       report={args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
