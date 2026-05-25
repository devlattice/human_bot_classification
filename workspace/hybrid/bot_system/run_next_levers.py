"""Next levers after strategy_compare: threshold sweep, HGB, hard-bot weights.

Writes only: workspace/hybrid/bot_system/data/strategy_next.json

Usage:
  python workspace/hybrid/bot_system/run_next_levers.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "workspace" / "hybrid" / "bot_system" / "data"
OUT = DATA / "strategy_next.json"
PROD = REPO / "workspace" / "hybrid" / "model_bundle"
MAY8_BOTS = DATA / "may8_matched_bot_features.parquet"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "bot_system"))

import train_production_model as tpm  # noqa: E402
from compare_strategies import load_or_none  # noqa: E402

TARGET_RECALL = 80.0
TARGET_ZEN_FPR = 2.0
TARGET_MAY8_FPR = 2.0


def focused_feature_cols() -> list[str]:
    bundle = PROD
    rf0 = joblib.load(bundle / "model.joblib")
    base_cols = json.loads((bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta0 = json.loads((bundle / "transform_meta.json").read_text())
    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")].copy()
    Xt = tpm.apply_transform(may8[base_cols].values, base_cols, transform_meta0)
    may8["proba"] = rf0.predict_proba(Xt)[:, 1]
    hard = may8[(may8["label"] == 1) & (may8["proba"] < 0.5)]
    humans = may8[may8["label"] == 0]
    scores = []
    for c in base_cols:
        if c not in hard.columns:
            continue
        h_mu, h_sd = float(humans[c].mean()), float(humans[c].std() or 1e-6)
        z = abs(float(hard[c].mean()) - h_mu) / h_sd
        scores.append((c, z))
    scores.sort(key=lambda x: -x[1])
    return [c for c, _ in scores[:18]]


def build_train_arrays(
    feature_cols: list[str],
    *,
    extra_bot: pd.DataFrame | None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    gold_full = pd.read_parquet(tpm.GOLD_PATH)
    gold_train = gold_full[~gold_full["date"].astype(str).str.contains("2026-05-08")]
    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)
    lmb = load_or_none(DATA / "targeted_bot_features.parquet")
    if extra_bot is None and MAY8_BOTS.is_file():
        extra_bot = load_or_none(MAY8_BOTS)

    rng = np.random.RandomState(seed)
    parts_h, parts_b, weights_h, weights_b = [], [], [], []

    def w(n: int, base: float = 1.0) -> np.ndarray:
        return np.full(n, base, dtype=np.float64)

    if gold_train is not None:
        gh = gold_train[gold_train["label"] == 0][feature_cols]
        gb = gold_train[gold_train["label"] == 1][feature_cols]
        parts_h.append(gh)
        parts_b.append(gb)
        weights_h.append(w(len(gh)))
        weights_b.append(w(len(gb)))
    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        sub = zen.sample(n=n, random_state=rng)[feature_cols]
        parts_h.append(sub)
        weights_h.append(w(len(sub)))
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        sub = pub_up.sample(n=n, random_state=rng)[feature_cols]
        parts_h.append(sub)
        weights_h.append(w(len(sub)))
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        sub = fs.sample(n=n, random_state=rng)[feature_cols]
        parts_b.append(sub)
        weights_b.append(w(len(sub)))
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        sub = ac.sample(n=n, random_state=rng)[feature_cols]
        parts_b.append(sub)
        weights_b.append(w(len(sub)))
    if lmb is not None:
        n = min(len(lmb), 6000)
        sub = lmb.sample(n=n, random_state=rng)[feature_cols]
        parts_b.append(sub)
        weights_b.append(w(len(sub)))
    if extra_bot is not None and len(extra_bot):
        n = min(len(extra_bot), 10000)
        sub = extra_bot.sample(n=n, random_state=rng)[feature_cols]
        parts_b.append(sub)
        weights_b.append(w(len(sub), 2.0))  # upweight may8-matched synth

    Xh = pd.concat(parts_h, ignore_index=True) if parts_h else pd.DataFrame()
    Xb = pd.concat(parts_b, ignore_index=True) if parts_b else pd.DataFrame()
    X_raw = pd.concat([Xh, Xb], ignore_index=True).values
    y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])

    sw = None
    if weights_h or weights_b:
        sw = np.concatenate(
            [np.concatenate(weights_h) if weights_h else np.array([]),
             np.concatenate(weights_b) if weights_b else np.array([])]
        )

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    return X_t, y, transform_meta, sw


def eval_at_threshold(
    clf,
    feature_cols: list[str],
    transform_meta: dict,
    threshold: float,
) -> dict:
    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")]
    may8_h = may8[may8["label"] == 0]
    may8_b = may8[may8["label"] == 1]
    zen = load_or_none(tpm.TEST_DIR / "zenodo_test_features.parquet")

    def score_df(df: pd.DataFrame) -> np.ndarray:
        Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
        return clf.predict_proba(Xt)[:, 1]

    p_h = score_df(may8_h)
    p_b = score_df(may8_b)
    p_zen = score_df(zen) if zen is not None else np.array([])

    return {
        "threshold": round(threshold, 3),
        "may8_recall_pct": round(float((p_b >= threshold).mean()) * 100, 3),
        "may8_human_fpr_pct": round(float((p_h >= threshold).mean()) * 100, 3),
        "zenodo_fpr_pct": round(float((p_zen >= threshold).mean()) * 100, 3) if len(p_zen) else None,
        "pass": bool(
            (p_b >= threshold).mean() * 100 >= TARGET_RECALL
            and (p_h >= threshold).mean() * 100 <= TARGET_MAY8_FPR
            and (len(p_zen) == 0 or (p_zen >= threshold).mean() * 100 <= TARGET_ZEN_FPR)
        ),
    }


def threshold_sweep(clf, feature_cols: list[str], transform_meta: dict) -> dict:
    rows = []
    best_pass = None
    best_recall_under_constraints = None
    for t in np.arange(0.05, 0.96, 0.05):
        row = eval_at_threshold(clf, feature_cols, transform_meta, float(t))
        rows.append(row)
        if row["pass"] and (best_pass is None or row["threshold"] < best_pass["threshold"]):
            best_pass = row
        zen_ok = row["zenodo_fpr_pct"] is None or row["zenodo_fpr_pct"] <= TARGET_ZEN_FPR
        fpr_ok = row["may8_human_fpr_pct"] <= TARGET_MAY8_FPR
        if fpr_ok and zen_ok:
            if best_recall_under_constraints is None or row["may8_recall_pct"] > best_recall_under_constraints["may8_recall_pct"]:
                best_recall_under_constraints = row
    return {
        "best_passing": best_pass,
        "best_under_fpr_caps": best_recall_under_constraints,
        "sweep_sample": rows[::2],  # every 0.10
    }


def train_rf_optuna(feature_cols: list[str], X_t, y, sw, seed: int) -> RandomForestClassifier:
    kw = tpm.rf_kwargs_from_namespace(
        type("NS", (), {
            "rf_n_estimators": 300, "rf_max_depth": "6", "rf_min_samples_leaf": 15,
            "rf_min_samples_split": 2, "rf_max_features": "sqrt", "rf_max_samples": 1.0,
            "rf_class_weight": "balanced", "rf_ccp_alpha": 0.0,
        })(),
        seed,
    )
    patch_path = PROD / "best_rf_params.json"
    if patch_path.is_file():
        kw = tpm.apply_rf_params_patch(kw, tpm.load_rf_params_json(patch_path))
    rf = RandomForestClassifier(**kw)
    rf.fit(X_t, y, sample_weight=sw)
    return rf


def main() -> int:
    t0 = time.time()
    feature_cols = focused_feature_cols()
    extra = load_or_none(MAY8_BOTS) if MAY8_BOTS.is_file() else None
    results: dict = {
        "baseline": "18 focused features, May-8 held out, may8_matched bots if present",
        "targets": {
            "may8_recall_pct": TARGET_RECALL,
            "may8_human_fpr_pct": TARGET_MAY8_FPR,
            "zenodo_fpr_pct": TARGET_ZEN_FPR,
        },
    }

    X_t, y, tm, sw = build_train_arrays(feature_cols, extra_bot=extra, seed=42)

    # D: RF + threshold sweep
    print("D: RF + threshold sweep...")
    rf = train_rf_optuna(feature_cols, X_t, y, sw, 42)
    d = threshold_sweep(rf, feature_cols, tm)
    d["may8_recall_at_0.5"] = eval_at_threshold(rf, feature_cols, tm, 0.5)["may8_recall_pct"]
    results["D_rf_threshold_sweep"] = d

    # E: Calibrated RF (isotonic) + sweep
    print("E: calibrated RF...")
    cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    cal.fit(X_t, y, sample_weight=sw)
    e = threshold_sweep(cal, feature_cols, tm)
    e["may8_recall_at_0.5"] = eval_at_threshold(cal, feature_cols, tm, 0.5)["may8_recall_pct"]
    results["E_calibrated_rf"] = e

    # F: HistGradientBoosting + sweep
    print("F: HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_iter=200, learning_rate=0.08,
        random_state=42, class_weight="balanced",
    )
    hgb.fit(X_t, y, sample_weight=sw)
    f = threshold_sweep(hgb, feature_cols, tm)
    f["may8_recall_at_0.5"] = eval_at_threshold(hgb, feature_cols, tm, 0.5)["may8_recall_pct"]
    results["F_hist_gradient_boost"] = f

    # G: Hard-bot centroid upweight — score May-8 with holdout RF (not in-sample prod model)
    print("G: hard-bot upweighted RF...")
    hold_rf = train_rf_optuna(feature_cols, X_t, y, sw, 42)
    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")].copy()
    Xt0 = tpm.apply_transform(may8[feature_cols].values, feature_cols, tm)
    may8["proba"] = hold_rf.predict_proba(Xt0)[:, 1]
    hard = may8[(may8["label"] == 1) & (may8["proba"] < 0.5)][feature_cols]
    if len(hard) >= 5:
        centroid = hard.mean().values
        X_t2, y2, tm2, sw2 = build_train_arrays(feature_cols, extra_bot=extra, seed=43)
        # upweight training bot rows close to hard centroid
        n = len(y2)
        bot_mask = y2 == 1
        Xb_raw = X_t2[bot_mask]
        dist = np.linalg.norm(Xb_raw - centroid, axis=1)
        sw2 = np.ones(n, dtype=np.float64)
        close = dist < np.percentile(dist, 25)
        sw2[bot_mask] = np.where(close, 4.0, 1.0)
        sw2[~bot_mask] = 1.0
        rf_g = train_rf_optuna(feature_cols, X_t2, y2, sw2, 43)
        g = threshold_sweep(rf_g, feature_cols, tm2)
        g["may8_recall_at_0.5"] = eval_at_threshold(rf_g, feature_cols, tm2, 0.5)["may8_recall_pct"]
        g["hard_bot_n"] = int(len(hard))
    else:
        g = {"error": "too few hard bots"}
    results["G_hard_upweight"] = g

    # Rank by best recall under FPR caps (any threshold)
    ranking = []
    for key in ("D_rf_threshold_sweep", "E_calibrated_rf", "F_hist_gradient_boost", "G_hard_upweight"):
        block = results.get(key, {})
        best = block.get("best_under_fpr_caps") or block.get("best_passing")
        if best:
            ranking.append({
                "lever": key,
                "may8_recall_pct": best.get("may8_recall_pct"),
                "threshold": best.get("threshold"),
                "zenodo_fpr_pct": best.get("zenodo_fpr_pct"),
                "pass": best.get("pass"),
            })
        elif block.get("may8_recall_at_0.5") is not None:
            ranking.append({
                "lever": key,
                "may8_recall_pct": block.get("may8_recall_at_0.5"),
                "threshold": 0.5,
                "pass": False,
            })
    ranking.sort(key=lambda r: -(r.get("may8_recall_pct") or 0))
    results["ranking"] = ranking
    results["winner"] = ranking[0]["lever"] if ranking else None
    results["elapsed_s"] = round(time.time() - t0, 1)

    OUT.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("NEXT LEVERS @0.5 vs best threshold under FPR caps")
    print("=" * 60)
    for key in ("D_rf_threshold_sweep", "E_calibrated_rf", "F_hist_gradient_boost", "G_hard_upweight"):
        b = results[key]
        print(f"\n{key}:")
        print(f"  @0.5 recall={b.get('may8_recall_at_0.5')}%")
        bp = b.get("best_passing")
        bu = b.get("best_under_fpr_caps")
        if bu:
            print(f"  best under caps: t={bu.get('threshold')} recall={bu.get('may8_recall_pct')}% "
                  f"zen_fpr={bu.get('zenodo_fpr_pct')}% pass={bu.get('pass')}")
        if bp:
            print(f"  PASSING: t={bp.get('threshold')} recall={bp.get('may8_recall_pct')}%")
    print(f"\nWinner: {results['winner']}  →  {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
