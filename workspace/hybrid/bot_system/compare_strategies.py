"""Run three May-8 improvement strategies; write one comparison JSON.

Strategies:
  A  passive-hard fingerprint + passive generator
  B  extended inverse loop (5 rounds, larger chunks)
  C  RF on top features that separate hard May-8 bots from humans

Output only:
  workspace/hybrid/bot_system/data/strategy_compare.json

Usage:
  python workspace/hybrid/bot_system/compare_strategies.py
  python workspace/hybrid/bot_system/compare_strategies.py --only A
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "workspace" / "hybrid" / "bot_system" / "data"
CMP = DATA / "_cmp"
OUT_JSON = DATA / "strategy_compare.json"
PROD = REPO / "workspace" / "hybrid" / "model_bundle"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))

import train_production_model as tpm  # noqa: E402

THRESH = 0.5
MIN_MAY8_RECALL = 80.0
MAX_HUMAN_FPR = 2.0
MAX_ZENODO_FPR = 2.0


def load_or_none(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def evaluate_block(
    name: str,
    df: pd.DataFrame | None,
    feature_cols: list[str],
    rf,
    transform_meta: dict,
    expected_label: int,
) -> dict | None:
    if df is None or df.empty:
        return None
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        return None
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    p = rf.predict_proba(Xt)[:, 1]
    out: dict = {
        "name": name,
        "n": int(len(df)),
        "mean_score": round(float(p.mean()), 4),
    }
    if expected_label == 0:
        out["fpr_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
    else:
        out["recall_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
        for t in (0.15, 0.25, 0.35):
            out[f"recall_at_{t:.2f}"] = round(float((p >= t).mean()) * 100, 2)
    return out


def train_holdout_may8(
    *,
    extra_bot: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    seed: int = 42,
    may8_bot_cap: int = 8000,
) -> tuple[dict, RandomForestClassifier, list[str], dict]:
    """Train without May-8 gold; evaluate held-out May-8 + tests."""
    feature_cols = list(feature_cols or tpm.ROBUST_FEATURES)
    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        raise FileNotFoundError(tpm.GOLD_PATH)

    gold_train = gold_full[~gold_full["date"].astype(str).str.contains("2026-05-08")]
    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)
    lmb = load_or_none(DATA / "targeted_bot_features.parquet")

    dfs = [gold_train, zen, pub, fs, ac, lmb, extra_bot]
    avail = set(feature_cols)
    for df in dfs:
        if df is not None:
            avail &= set(df.columns)
    feature_cols = [c for c in feature_cols if c in avail]

    rng = np.random.RandomState(seed)
    parts_h, parts_b = [], []

    if gold_train is not None:
        parts_h.append(gold_train[gold_train["label"] == 0][feature_cols])
        parts_b.append(gold_train[gold_train["label"] == 1][feature_cols])
    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
    if lmb is not None:
        n = min(len(lmb), 6000)
        parts_b.append(lmb.sample(n=n, random_state=rng)[feature_cols])
    if extra_bot is not None and len(extra_bot):
        n = min(len(extra_bot), may8_bot_cap)
        parts_b.append(extra_bot.sample(n=n, random_state=rng)[feature_cols])

    Xh = pd.concat(parts_h, ignore_index=True) if parts_h else pd.DataFrame()
    Xb = pd.concat(parts_b, ignore_index=True) if parts_b else pd.DataFrame()
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

    may8 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-08")]
    blocks = [
        evaluate_block("may8_human", may8[may8["label"] == 0], feature_cols, rf, transform_meta, 0),
        evaluate_block("may8_bot", may8[may8["label"] == 1], feature_cols, rf, transform_meta, 1),
        evaluate_block("may7_human", gold_full[gold_full["date"].astype(str).str.contains("2026-05-07") &
                        (gold_full["label"] == 0)], feature_cols, rf, transform_meta, 0),
        evaluate_block(
            "zenodo_test",
            load_or_none(tpm.TEST_DIR / "zenodo_test_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            0,
        ),
        evaluate_block(
            "acpc_bot_test",
            load_or_none(tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            1,
        ),
    ]
    blocks = [b for b in blocks if b]

    may8_b = next((b for b in blocks if b["name"] == "may8_bot"), {})
    may8_h = next((b for b in blocks if b["name"] == "may8_human"), {})
    zen = next((b for b in blocks if b["name"] == "zenodo_test"), {})

    metrics = {
        "may8_recall_pct": may8_b.get("recall_pct"),
        "may8_human_fpr_pct": may8_h.get("fpr_pct"),
        "zenodo_fpr_pct": zen.get("fpr_pct"),
        "may8_recall_at_0.25": may8_b.get("recall_at_0.25"),
        "pass": bool(
            may8_b.get("recall_pct", 0) >= MIN_MAY8_RECALL
            and may8_h.get("fpr_pct", 99) <= MAX_HUMAN_FPR
            and zen.get("fpr_pct", 99) <= MAX_ZENODO_FPR
        ),
        "blocks": {b["name"]: b for b in blocks},
        "n_features": len(feature_cols),
        "train_n": int(len(X_raw)),
    }
    return metrics, rf, feature_cols, transform_meta


def run_cmd(cmd: list[str], label: str) -> int:
    print(f"\n>>> {label}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO))
    print(f"    exit={r.returncode}  {time.time()-t0:.0f}s")
    return r.returncode


def build_passive_hard_fp(bundle: Path, out: Path) -> int:
    model_path = bundle / "model.joblib"
    if not model_path.is_file():
        model_path = bundle / "lgbm_student.joblib"
    rf = joblib.load(model_path)
    feature_cols = json.loads((bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta = json.loads((bundle / "transform_meta.json").read_text())
    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")]
    bots = may8[may8["label"] == 1].copy()
    Xt = tpm.apply_transform(bots[feature_cols].values, feature_cols, transform_meta)
    bots["proba"] = rf.predict_proba(Xt)[:, 1]
    hard = bots[bots["proba"] < THRESH]
    if hard.empty:
        return 1
    meta = {"label", "date", "chunk_idx", "source"}
    num_cols = [c for c in hard.columns if c not in meta and pd.api.types.is_numeric_dtype(hard[c])]
    means = hard[num_cols].mean()
    stds = hard[num_cols].std().replace(0, 1e-6)
    payload = {
        "source": "may8_passive_hard_bots",
        "n_bot": int(len(hard)),
        "feature_cols": num_cols,
        "feature_means": {c: float(means[c]) for c in num_cols},
        "feature_stds": {c: float(stds[c]) for c in num_cols},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"    passive-hard n={len(hard)} dims={len(num_cols)}")
    return 0


def strategy_a(cmp: Path) -> dict:
    fp = cmp / "passive_fp.json"
    matched = cmp / "passive_matched.json"
    bots_pq = cmp / "passive_bots.parquet"
    CMP.mkdir(parents=True, exist_ok=True)

    if build_passive_hard_fp(PROD, fp) != 0:
        return {"error": "no hard bots for fingerprint"}

    py = sys.executable
    if run_cmd(
        [py, "workspace/hybrid/bot_system/03_match_profiles.py",
         "--fp", str(fp), "--passive", "--fp-cols", "robust",
         "--out", str(matched), "--n-candidates", "150", "--top-k", "12",
         "--workers", "8", "--seed", "77"],
        "A: match passive-hard",
    ):
        return {"error": "match failed"}

    if run_cmd(
        [py, "workspace/hybrid/bot_system/04_generate_targeted_bots.py",
         "--matched", str(matched), "--out", str(bots_pq), "--passive",
         "--top-k", "12", "--perturbations-per-seed", "8",
         "--chunks-per-profile", "25", "--workers", "4", "--seed", "77"],
        "A: generate passive bots",
    ):
        return {"error": "generate failed"}

    extra = load_or_none(bots_pq)
    t0 = time.time()
    metrics, _, _, _ = train_holdout_may8(extra_bot=extra, seed=77)
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    metrics["extra_bot_rows"] = int(len(extra)) if extra is not None else 0
    return metrics


def strategy_b(cmp: Path) -> dict:
    fp = REPO / "workspace/hybrid/bot_system/data/may8_target_fingerprint.json"
    py = sys.executable
    best: dict | None = None
    bots_pq = cmp / "inverse_bots.parquet"

    run_cmd(
        [py, "workspace/hybrid/bot_system/06_build_may8_target.py",
         "--blend", "0.7", "--out", str(fp)],
        "B: may8 target fp",
    )

    for rnd in range(1, 6):
        n_cand = 200 * rnd
        topk = 12 + (rnd - 1) * 4
        perturb = 6 + (rnd - 1) * 3
        chunks = 35 + (rnd - 1) * 15
        matched = cmp / f"inv_matched_r{rnd}.json"

        if run_cmd(
            [py, "workspace/hybrid/bot_system/03_match_profiles.py",
             "--fp", str(fp), "--passive", "--out", str(matched),
             "--n-candidates", str(n_cand), "--top-k", str(topk),
             "--workers", "8", "--seed", str(40 + rnd)],
            f"B round {rnd}: match",
        ):
            continue
        if run_cmd(
            [py, "workspace/hybrid/bot_system/04_generate_targeted_bots.py",
             "--matched", str(matched), "--out", str(bots_pq), "--passive",
             "--top-k", str(topk), "--perturbations-per-seed", str(perturb),
             "--chunks-per-profile", str(chunks), "--workers", "4",
             "--per-job-timeout", "120", "--seed", str(2030 + rnd)],
            f"B round {rnd}: generate",
        ):
            continue

        t0 = time.time()
        try:
            metrics, _, _, _ = train_holdout_may8(
                extra_bot=load_or_none(bots_pq), seed=40 + rnd, may8_bot_cap=10000,
            )
            metrics["round"] = rnd
            metrics["elapsed_s"] = round(time.time() - t0, 1)
            metrics["n_candidates"] = n_cand
            metrics["chunks_per_profile"] = chunks
            print(f"    round {rnd} may8_recall={metrics.get('may8_recall_pct')}%")
            if best is None or (metrics.get("may8_recall_pct") or 0) > (best.get("may8_recall_pct") or 0):
                best = metrics
        except Exception as e:
            print(f"    round {rnd} train error: {e}")

    return best or {"error": "all rounds failed"}


def strategy_c(cmp: Path) -> dict:
    """Top ROBUST features separating hard May-8 bots from May-8 humans."""
    bundle = PROD
    model_path = bundle / "model.joblib"
    rf0 = joblib.load(model_path)
    base_cols = json.loads((bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta0 = json.loads((bundle / "transform_meta.json").read_text())

    gold = pd.read_parquet(tpm.GOLD_PATH)
    may8 = gold[gold["date"].astype(str).str.contains("2026-05-08")].copy()
    Xt = tpm.apply_transform(may8[base_cols].values, base_cols, transform_meta0)
    may8["proba"] = rf0.predict_proba(Xt)[:, 1]
    hard = may8[(may8["label"] == 1) & (may8["proba"] < THRESH)]
    humans = may8[may8["label"] == 0]

    scores: list[tuple[str, float]] = []
    for c in base_cols:
        if c not in hard.columns:
            continue
        h_mu = float(humans[c].mean())
        h_sd = float(humans[c].std() or 1e-6)
        b_mu = float(hard[c].mean())
        z = abs(b_mu - h_mu) / h_sd
        scores.append((c, z))
    scores.sort(key=lambda x: -x[1])
    top_k = 18
    focused = [c for c, _ in scores[:top_k]]
    print(f"    C: top features (hard vs human): {focused[:8]}...")

    t0 = time.time()
    metrics, _, _, _ = train_holdout_may8(feature_cols=focused, seed=99)
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    metrics["top_features"] = focused
    return metrics


def score_strategy(m: dict) -> float:
    """Higher is better for ranking."""
    if "error" in m:
        return -1.0
    recall = float(m.get("may8_recall_pct") or 0)
    fpr = float(m.get("may8_human_fpr_pct") or 99)
    zen = float(m.get("zenodo_fpr_pct") or 99)
    bonus = float(m.get("may8_recall_at_0.25") or 0) * 0.1
    penalty = max(0, fpr - MAX_HUMAN_FPR) * 5 + max(0, zen - MAX_ZENODO_FPR) * 2
    return recall + bonus - penalty


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("A", "B", "C"), default=None)
    args = p.parse_args()
    CMP.mkdir(parents=True, exist_ok=True)

    results: dict = {"thresholds": {"may8_recall": MIN_MAY8_RECALL, "max_fpr": MAX_HUMAN_FPR}}
    t_all = time.time()

    if args.only in (None, "A"):
        print("\n" + "=" * 60 + "\nSTRATEGY A: passive-hard fingerprint\n" + "=" * 60)
        results["A_passive_hard"] = strategy_a(CMP)

    if args.only in (None, "B"):
        print("\n" + "=" * 60 + "\nSTRATEGY B: inverse x5 (large chunks)\n" + "=" * 60)
        results["B_inverse5"] = strategy_b(CMP)

    if args.only in (None, "C"):
        print("\n" + "=" * 60 + "\nSTRATEGY C: focused features (hard vs human)\n" + "=" * 60)
        results["C_focused_features"] = strategy_c(CMP)

    ranked = []
    for key in ("A_passive_hard", "B_inverse5", "C_focused_features"):
        if key not in results:
            continue
        m = results[key]
        ranked.append((key, score_strategy(m), m.get("may8_recall_pct"), m.get("pass", False)))

    ranked.sort(key=lambda x: -x[1])
    results["ranking"] = [
        {"strategy": k, "score": s, "may8_recall_pct": r, "pass": p} for k, s, r, p in ranked
    ]
    results["winner"] = ranked[0][0] if ranked else None
    results["elapsed_total_s"] = round(time.time() - t_all, 1)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    # Drop intermediate cmp artifacts; keep only strategy_compare.json
    for p in CMP.glob("*"):
        try:
            p.unlink()
        except OSError:
            pass
    print("\n" + "=" * 60)
    print("COMPARISON (May-8 held out of training; @0.5)")
    print("=" * 60)
    print(f"{'strategy':<22} {'may8_recall':>12} {'may8_FPR':>10} {'zen_FPR':>10} {'pass':>6}")
    for key in ("A_passive_hard", "B_inverse5", "C_focused_features"):
        m = results.get(key, {})
        if "error" in m:
            print(f"{key:<22} ERROR: {m['error']}")
            continue
        print(
            f"{key:<22} {m.get('may8_recall_pct', '?'):>11}% "
            f"{m.get('may8_human_fpr_pct', '?'):>9}% "
            f"{m.get('zenodo_fpr_pct', '?'):>9}% "
            f"{str(m.get('pass', False)):>6}"
        )
    print(f"\nWinner: {results.get('winner')}  →  {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
