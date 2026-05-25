"""Train a model that **holds May-8 gold completely out of training**, mix in
the freshly generated May-8-matched bots, and report whether the four pass
criteria are satisfied:

    1. May-8 bot recall          >= MIN_MAY8_RECALL   (default 80%)
    2. May-8 human FPR           <= MAX_HUMAN_FPR     (default 2%)
    3. May-7 human FPR           <= MAX_HUMAN_FPR     (regression guard)
    4. Zenodo-test human FPR     <= MAX_ZENODO_FPR    (default 2%)

Result is written to:
    workspace/hybrid/bot_system/data/inverse_loop_round_<idx>.json
and the model bundle is saved to:
    workspace/hybrid/bot_system/data/round_<idx>_bundle/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid" / "scripts"))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

import train_production_model as tpm  # type: ignore  # noqa: E402

DATA_DIR = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data"
DEFAULT_MAY8_MATCHED = DATA_DIR / "may8_matched_bot_features.parquet"
DEFAULT_LIVE_MATCHED = DATA_DIR / "targeted_bot_features.parquet"


# ---------------- helpers ----------------------------------------------------

def safe_concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and len(p) > 0]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_or_none(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def evaluate_block(name: str, df: pd.DataFrame | None,
                   feature_cols: list[str], rf,
                   transform_meta: dict, expected_label: int) -> dict | None:
    if df is None or df.empty:
        return None
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {"name": name, "error": f"missing {len(missing)} cols", "n": int(len(df))}
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    proba = rf.predict_proba(Xt)[:, 1]
    out = {
        "name": name,
        "n": int(len(df)),
        "mean_score": round(float(proba.mean()), 4),
        "median_score": round(float(np.median(proba)), 4),
    }
    if expected_label == 0:
        out["fpr_pct"] = round(float((proba >= 0.5).mean()) * 100, 3)
    else:
        out["recall_pct"] = round(float((proba >= 0.5).mean()) * 100, 3)
    return out


# ---------------- main -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--may8-matched", type=Path, default=DEFAULT_MAY8_MATCHED)
    p.add_argument("--live-matched", type=Path, default=DEFAULT_LIVE_MATCHED)
    p.add_argument("--may8-bot-cap", type=int, default=8000,
                   help="Max chunks to sample from the May-8 matched set.")
    p.add_argument("--min-may8-recall", type=float, default=80.0)
    p.add_argument("--max-human-fpr", type=float, default=2.0)
    p.add_argument("--max-zenodo-fpr", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print(f"INVERSE LOOP — ROUND {args.round}")
    print("=" * 70)
    print(f"may8_matched={args.may8_matched}")
    print(f"live_matched={args.live_matched}")

    feature_cols = list(tpm.ROBUST_FEATURES)

    # ── Training data: deliberately exclude May-8 gold ──────────────────────
    print("\n[load] training corpora (May-8 gold EXCLUDED)")
    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        print(f"[error] missing gold features: {tpm.GOLD_PATH}")
        return 2
    gold_train = gold_full
    if "date" in gold_full.columns and gold_full["date"].astype(str).str.contains("2026-05-08").any():
        gold_train = gold_full[~gold_full["date"].str.contains("2026-05-08")]
        print(f"  gold_train rows: {len(gold_train)} "
              f"(dropped {len(gold_full) - len(gold_train)} may-8 rows from train parquet)")
    else:
        print(f"  gold_train rows: {len(gold_train)} (May-8 not in train parquet)")

    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    gb = load_or_none(tpm.GEN_BOT_PATH)
    cb = load_or_none(tpm.CAL_BOT_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)
    lmb = load_or_none(args.live_matched)
    mmb = load_or_none(args.may8_matched)

    available = [df for df in [gold_train, zen, pub, fs, gb, cb, ac, lmb, mmb]
                 if df is not None]
    avail_feats = set(feature_cols)
    for df in available:
        avail_feats &= set(df.columns)
    feature_cols = [c for c in feature_cols if c in avail_feats]
    print(f"[features] common to all datasets: {len(feature_cols)}")

    rng = np.random.RandomState(args.seed)
    parts_h, parts_b = [], []
    src_meta: dict[str, int] = {}

    if gold_train is not None:
        parts_h.append(gold_train[gold_train["label"] == 0][feature_cols])
        parts_b.append(gold_train[gold_train["label"] == 1][feature_cols])
        src_meta["gold_human"] = int((gold_train["label"] == 0).sum())
        src_meta["gold_bot"] = int((gold_train["label"] == 1).sum())

    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
        src_meta["zenodo_human"] = n
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        src_meta["public_human"] = n
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
        src_meta["full_spectrum_bot"] = n
    if gb is not None:
        n = min(len(gb), tpm.BOT_SAMPLE_CAP // 2)
        parts_b.append(gb.sample(n=n, random_state=rng)[feature_cols])
        src_meta["gen_bot"] = n
    if cb is not None:
        n = min(len(cb), tpm.BOT_SAMPLE_CAP // 2)
        parts_b.append(cb.sample(n=n, random_state=rng)[feature_cols])
        src_meta["cal_bot"] = n
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
        src_meta["acpc_bot"] = n
    if lmb is not None:
        n = min(len(lmb), 6000)
        parts_b.append(lmb.sample(n=n, random_state=rng)[feature_cols])
        src_meta["live_matched_bot"] = n
    if mmb is not None:
        n = min(len(mmb), args.may8_bot_cap)
        parts_b.append(mmb.sample(n=n, random_state=rng)[feature_cols])
        src_meta["may8_matched_bot"] = n

    Xh = safe_concat(parts_h)
    Xb = safe_concat(parts_b)
    X_raw = pd.concat([Xh, Xb], ignore_index=True).values
    y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])
    print(f"[mix] human={len(Xh)} bot={len(Xb)} total={len(X_raw)} "
          f"sources={src_meta}")

    # ── Fit transforms + train ──────────────────────────────────────────────
    print("\n[transform] fitting clip + log1p + robust scale")
    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)

    print("[train] RandomForest …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=args.seed, n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_t, y)
    train_proba = rf.predict_proba(X_t)[:, 1]
    train_auc = float(roc_auc_score(y, train_proba))
    train_acc = float(accuracy_score(y, (train_proba >= 0.5).astype(int)))
    print(f"  done in {time.time()-t0:.1f}s  train AUC={train_auc:.4f} "
          f"acc={train_acc:.4f}")

    # ── Held-out evaluation ─────────────────────────────────────────────────
    may8 = load_or_none(tpm.MAY8_GOLD_TEST_PATH)
    if may8 is None or may8.empty:
        may8 = gold_full[gold_full["date"].str.contains("2026-05-08")]
    may7 = gold_full[gold_full["date"].str.contains("2026-05-07")]
    may8_h = may8[may8["label"] == 0]
    may8_b = may8[may8["label"] == 1]
    may7_h = may7[may7["label"] == 0]
    may7_b = may7[may7["label"] == 1]

    res_may8_h = evaluate_block("may8_human", may8_h, feature_cols, rf, transform_meta, 0)
    res_may8_b = evaluate_block("may8_bot", may8_b, feature_cols, rf, transform_meta, 1)
    res_may7_h = evaluate_block("may7_human", may7_h, feature_cols, rf, transform_meta, 0)
    res_may7_b = evaluate_block("may7_bot", may7_b, feature_cols, rf, transform_meta, 1)

    res_zen_test = evaluate_block(
        "zenodo_test",
        load_or_none(tpm.TEST_DIR / "zenodo_test_features.parquet"),
        feature_cols, rf, transform_meta, 0,
    )
    res_pub_test = evaluate_block(
        "public_test",
        load_or_none(tpm.TEST_DIR / "public_test_features.parquet"),
        feature_cols, rf, transform_meta, 0,
    )
    res_acpc_test = evaluate_block(
        "acpc_bot_test",
        load_or_none(tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
        feature_cols, rf, transform_meta, 1,
    )

    blocks = [r for r in [res_may8_h, res_may8_b, res_may7_h, res_may7_b,
                          res_zen_test, res_pub_test, res_acpc_test] if r]

    print("\n" + "=" * 70)
    print("HELD-OUT EVALUATION")
    print("=" * 70)
    for r in blocks:
        if "fpr_pct" in r:
            print(f"  {r['name']:18s} n={r['n']:5d}  FPR={r['fpr_pct']:.3f}%  "
                  f"mean={r['mean_score']:.4f}")
        elif "recall_pct" in r:
            print(f"  {r['name']:18s} n={r['n']:5d}  recall={r['recall_pct']:.3f}%  "
                  f"mean={r['mean_score']:.4f}")

    # ── Pass / fail ─────────────────────────────────────────────────────────
    pass_status = {
        "may8_recall_pct": res_may8_b["recall_pct"] if res_may8_b else None,
        "may8_human_fpr_pct": res_may8_h["fpr_pct"] if res_may8_h else None,
        "may7_human_fpr_pct": res_may7_h["fpr_pct"] if res_may7_h else None,
        "zenodo_fpr_pct": res_zen_test["fpr_pct"] if res_zen_test else None,
    }
    def _ge(v, lo):
        return v is not None and v >= lo

    def _le(v, hi):
        return v is not None and v <= hi

    checks = {
        "may8_recall_ok": _ge(pass_status["may8_recall_pct"], args.min_may8_recall),
        "may8_human_fpr_ok": _le(pass_status["may8_human_fpr_pct"], args.max_human_fpr),
        "may7_human_fpr_ok": _le(pass_status["may7_human_fpr_pct"], args.max_human_fpr),
        "zenodo_fpr_ok": _le(pass_status["zenodo_fpr_pct"], args.max_zenodo_fpr),
    }
    overall_pass = all(checks.values())

    print("\n" + "=" * 70)
    print("PASS CRITERIA")
    print("=" * 70)
    print(f"  may8 recall    >= {args.min_may8_recall:.1f}%  "
          f"actual={pass_status['may8_recall_pct']}  {'OK' if checks['may8_recall_ok'] else 'FAIL'}")
    print(f"  may8 FPR       <= {args.max_human_fpr:.1f}%  "
          f"actual={pass_status['may8_human_fpr_pct']}  {'OK' if checks['may8_human_fpr_ok'] else 'FAIL'}")
    print(f"  may7 FPR       <= {args.max_human_fpr:.1f}%  "
          f"actual={pass_status['may7_human_fpr_pct']}  {'OK' if checks['may7_human_fpr_ok'] else 'FAIL'}")
    print(f"  zenodo FPR     <= {args.max_zenodo_fpr:.1f}%  "
          f"actual={pass_status['zenodo_fpr_pct']}  {'OK' if checks['zenodo_fpr_ok'] else 'FAIL'}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # ── Persist round artifacts ─────────────────────────────────────────────
    round_dir = args.out_dir / f"round_{args.round:02d}_bundle"
    round_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(rf, round_dir / "lgbm_student.joblib")
    (round_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2)
    )
    (round_dir / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2)
    )

    summary = {
        "round": args.round,
        "ts": datetime.utcnow().isoformat() + "Z",
        "feature_cols_n": len(feature_cols),
        "sources": src_meta,
        "train_auc": round(train_auc, 6),
        "train_acc": round(train_acc, 6),
        "evaluations": {b["name"]: b for b in blocks},
        "pass_status": pass_status,
        "checks": checks,
        "overall_pass": bool(overall_pass),
        "thresholds": {
            "min_may8_recall_pct": args.min_may8_recall,
            "max_human_fpr_pct": args.max_human_fpr,
            "max_zenodo_fpr_pct": args.max_zenodo_fpr,
        },
    }
    out_path = args.out_dir / f"inverse_loop_round_{args.round:02d}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {out_path}\n       bundle={round_dir}")
    return 0 if overall_pass else 10  # 10 = soft-fail (caller may iterate)


if __name__ == "__main__":
    raise SystemExit(main())
