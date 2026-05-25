"""Sweep decision thresholds on May-8 hold-out and test FPR for a model bundle.

Usage:
  python workspace/hybrid/bot_system/25_may8_threshold_sweep.py \\
    --bundle workspace/hybrid/model_bundle_may8_hard_focus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))

import train_production_model as tpm  # noqa: E402

MAY8_DATE = "2026-05-08"
THRESHOLDS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_bundle(bundle: Path) -> tuple[RandomForestClassifier, list[str], dict]:
    rf = joblib.load(bundle / "lgbm_student.joblib")
    meta = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))
    tm = json.loads((bundle / "transform_meta.json").read_text(encoding="utf-8"))
    return rf, meta["feature_cols"], tm


def scores(df: pd.DataFrame, cols: list[str], rf: RandomForestClassifier, tm: dict) -> np.ndarray:
    Xt = tpm.apply_transform(df[cols].values, cols, tm)
    return rf.predict_proba(Xt)[:, 1]


def recall_at(p: np.ndarray, t: float) -> float:
    return float((p >= t).mean()) * 100


def may8_hard_easy(
    may8_b: pd.DataFrame, cols: list[str], gold_train: pd.DataFrame, zen: pd.DataFrame | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    ref_h = [gold_train[gold_train["label"] == 0][cols]]
    ref_b = [gold_train[gold_train["label"] == 1][cols]]
    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        ref_h.append(zen.sample(n=n, random_state=seed)[cols])
    Xrh = pd.concat(ref_h, ignore_index=True)
    Xrb = pd.concat(ref_b, ignore_index=True)
    Xr = pd.concat([Xrh, Xrb], ignore_index=True).values
    yr = np.concatenate([np.zeros(len(Xrh)), np.ones(len(Xrb))])
    Xrt, ref_tm = tpm.fit_transform_pipeline(Xr, cols)
    ref_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    ref_rf.fit(Xrt, yr)
    ref_p = ref_rf.predict_proba(
        tpm.apply_transform(may8_b[cols].values, cols, ref_tm)
    )[:, 1]
    hard_mask = ref_p < 0.5
    return hard_mask, ~hard_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle
    rf, cols, tm = load_bundle(bundle)

    gold = pd.read_parquet(tpm.GOLD_PATH)
    gold_train = gold[~gold["date"].astype(str).str.contains(MAY8_DATE)]
    may8 = gold[gold["date"].astype(str).str.contains(MAY8_DATE)]
    may8_h = may8[may8["label"] == 0]
    may8_b = may8[may8["label"] == 1]
    zen = pd.read_parquet(tpm.ZENODO_PATH) if tpm.ZENODO_PATH.is_file() else None
    zen_test = pd.read_parquet(tpm.TEST_DIR / "zenodo_test_features.parquet")

    hard_mask, easy_mask = may8_hard_easy(may8_b, cols, gold_train, zen, args.seed)
    p_m8b = scores(may8_b, cols, rf, tm)
    p_m8h = scores(may8_h, cols, rf, tm)
    p_hard = p_m8b[hard_mask]
    p_easy = p_m8b[easy_mask]
    p_zen = scores(zen_test, cols, rf, tm)

    rows = []
    for t in THRESHOLDS:
        rows.append({
            "threshold": t,
            "may8_bot_recall_pct": round(recall_at(p_m8b, t), 2),
            "may8_hard_recall_pct": round(recall_at(p_hard, t), 2) if len(p_hard) else None,
            "may8_easy_recall_pct": round(recall_at(p_easy, t), 2) if len(p_easy) else None,
            "may8_human_fpr_pct": round(recall_at(p_m8h, t), 3),
            "zenodo_fpr_pct": round(recall_at(p_zen, t), 3),
        })

    print(f"\n=== Threshold sweep: {bundle.name} ===")
    print(f"{'t':>5}  {'m8_bot':>8}  {'m8_hard':>8}  {'m8_easy':>8}  {'m8_hFPR':>8}  {'zen_FPR':>8}")
    for r in rows:
        print(
            f"{r['threshold']:5.2f}  {r['may8_bot_recall_pct']:8.2f}  "
            f"{r['may8_hard_recall_pct'] or 0:8.2f}  {r['may8_easy_recall_pct'] or 0:8.2f}  "
            f"{r['may8_human_fpr_pct']:8.3f}  {r['zenodo_fpr_pct']:8.3f}"
        )

  # best t for hard recall >= 80 with zen FPR <= 2
    best = None
    for r in rows:
        if (r["may8_hard_recall_pct"] or 0) >= 80 and (r["zenodo_fpr_pct"] or 100) <= 2.0:
            best = r
    if best:
        print(f"\n[hint] hard≥80% & zen≤2%: threshold={best['threshold']} "
              f"(bot={best['may8_bot_recall_pct']}%, hard={best['may8_hard_recall_pct']}%)")
    else:
        print("\n[hint] no threshold hits hard≥80% with zenodo FPR≤2% on this bundle")

    out = args.out or (
        REPO / "workspace" / "hybrid" / "bot_system" / "data" / f"threshold_sweep_{bundle.name}.json"
    )
    out.write_text(json.dumps({"bundle": str(bundle), "sweep": rows}, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
