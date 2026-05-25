"""Diagnose why ~64% of May-8 gold bots stay below the 0.5 threshold.

Loads the latest round bundle (or a specified one), scores all May-8 gold,
splits bots into 'easy' (>=0.5) vs 'hard' (<0.5), then:

1. Reports score histogram of bots vs humans on May-8.
2. Compares feature means of hard vs easy bots (top 25 by |z|).
3. Compares hard bots to:
     - May-8 humans   (do hard bots look like humans?)
     - All training bots (do hard bots fall outside the bot manifold?)
4. Reports basic chunk-level stats (n_actions, n_players, etc).

Outputs:
    workspace/hybrid/bot_system/data/may8_hard_diagnosis.txt
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
import train_production_model as tpm  # type: ignore  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "round_03_bundle"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "may8_hard_diagnosis.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def fmt(x, fmt_="%.4f"):
    try:
        return fmt_ % x
    except Exception:
        return str(x)


def main() -> int:
    args = parse_args()
    bundle = args.bundle
    rf = joblib.load(bundle / "lgbm_student.joblib")
    feature_cols = json.loads((bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta = json.loads((bundle / "transform_meta.json").read_text())
    lines: list[str] = []

    def out(msg: str = ""):
        print(msg)
        lines.append(msg)

    out(f"[bundle] {bundle}")
    out(f"[features] {len(feature_cols)}")

    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].str.contains("2026-05-08")].copy()
    out(f"[may8] n={len(may8)}  bots={(may8['label']==1).sum()}  "
        f"humans={(may8['label']==0).sum()}")

    X = tpm.apply_transform(may8[feature_cols].values, feature_cols, transform_meta)
    proba = rf.predict_proba(X)[:, 1]
    may8["proba"] = proba

    bots = may8[may8["label"] == 1].copy()
    humans = may8[may8["label"] == 0].copy()
    easy = bots[bots["proba"] >= 0.5]
    hard = bots[bots["proba"] < 0.5]
    out("\n=== May-8 score distribution ===")
    out(f"  bots:  n={len(bots)}  mean={bots['proba'].mean():.4f}  "
        f"median={bots['proba'].median():.4f}")
    out(f"    >=0.50  {len(easy):3d}  ({100*len(easy)/len(bots):.1f}%)")
    out(f"    <0.50   {len(hard):3d}  ({100*len(hard)/len(bots):.1f}%)")
    for q in (10, 25, 50, 75, 90, 95):
        out(f"    p{q:02d}     {np.percentile(bots['proba'], q):.4f}")
    out(f"  humans: n={len(humans)}  mean={humans['proba'].mean():.4f}  "
        f"max={humans['proba'].max():.4f}")
    # Score distribution of hard cluster
    out("\n  hard-bot score percentiles:")
    for q in (10, 25, 50, 75, 90, 95):
        out(f"    p{q:02d} = {np.percentile(hard['proba'], q):.4f}")

    # ── How distinct are hard and easy bots in raw features? ───────────
    out("\n=== Hard vs Easy bots: top features by |z| ===")
    rows = []
    for c in feature_cols:
        if c not in hard.columns:
            continue
        h_mean = float(hard[c].mean())
        h_sd = float(hard[c].std() or 1e-6)
        e_mean = float(easy[c].mean()) if len(easy) else 0.0
        z = abs((e_mean - h_mean) / h_sd) if h_sd else 0.0
        rows.append((c, z, h_mean, e_mean))
    rows.sort(key=lambda r: -r[1])
    out(f"  {'feature':38s}  {'|z|':>6s}  {'hard_mu':>10s}  {'easy_mu':>10s}")
    for c, z, hm, em in rows[:25]:
        out(f"  {c:38s}  {z:6.2f}  {fmt(hm,'%10.4f')}  {fmt(em,'%10.4f')}")

    # ── Do hard bots look like May-8 humans? ─────────────────────────────
    out("\n=== Hard bots vs May-8 humans: top features by |z| ===")
    rows = []
    for c in feature_cols:
        if c not in humans.columns:
            continue
        h_mu = float(humans[c].mean())
        h_sd = float(humans[c].std() or 1e-6)
        b_mu = float(hard[c].mean())
        z = abs((b_mu - h_mu) / h_sd) if h_sd else 0.0
        rows.append((c, z, h_mu, b_mu))
    rows.sort(key=lambda r: -r[1])
    out(f"  {'feature':38s}  {'|z|':>6s}  {'hum_mu':>10s}  {'hard_mu':>10s}")
    for c, z, hm, bm in rows[:25]:
        out(f"  {c:38s}  {z:6.2f}  {fmt(hm,'%10.4f')}  {fmt(bm,'%10.4f')}")

    # ── Basic chunk-level stats: are hard bots a different game shape? ──
    out("\n=== Basic stats (raw, before transform) ===")
    cols_basic = [c for c in ["n_players_mean", "n_streets_mean", "n_actions_mean",
                              "mean_pot_after_mean", "bet_size_mean_mean", "chunk_n_hands"]
                  if c in may8.columns]
    if cols_basic:
        groups = {
            "easy_bot": easy,
            "hard_bot": hard,
            "may8_human": humans,
        }
        out(f"  {'feature':24s}  " + "  ".join(f"{k:>14s}" for k in groups))
        for c in cols_basic:
            vals = [f"{groups[k][c].mean():14.4f}" for k in groups]
            out(f"  {c:24s}  " + "  ".join(vals))

    # ── Does score correlate with anything obvious? ─────────────────────
    out("\n=== Score correlation with raw features (top 15 |corr|) ===")
    rows = []
    for c in feature_cols:
        if c not in bots.columns:
            continue
        v = bots[c].fillna(0.0).values
        if v.std() < 1e-9:
            continue
        corr = float(np.corrcoef(v, bots["proba"].values)[0, 1])
        rows.append((c, abs(corr), corr))
    rows.sort(key=lambda r: -r[1])
    for c, _, r in rows[:15]:
        out(f"  {c:38s}  corr={r:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
