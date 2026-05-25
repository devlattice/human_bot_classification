"""Hard-bot-focused retrain: gold (May-8 held out) + heavy Phase-3 synthetic mix.

Oversamples ``may8_reflect_bot_features.parquet`` with per-row sample weights so the
RF sees more May-8-shaped bot mass without leaking May-8 labels into training.

Evaluates:
  - zenodo_test, public_test, acpc_bot_test, wsop_stress (held-out test parquets)
  - May-8 gold (human FPR, bot recall, hard/easy split via reference hold-out RF)
  - unlabeled real_distribution JSONL

Usage:
  python workspace/hybrid/bot_system/24_retrain_may8_hard_focus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

import train_production_model as tpm  # noqa: E402
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402

DATA = REPO / "workspace" / "hybrid" / "bot_system" / "data"
DEFAULT_SYN = DATA / "may8_reflect_bot_features.parquet"
DEFAULT_BUNDLE = REPO / "workspace" / "hybrid" / "model_bundle_may8_hard_focus"
DEFAULT_OUT = DATA / "may8_hard_focus_eval_results.json"
REAL_DIST = REPO / "workspace" / "dataset" / "real_distribution"
THRESH = 0.5
MAY8_DATE = "2026-05-08"


def load_or_none(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def evaluate_block(
    name: str,
    df: pd.DataFrame | None,
    feature_cols: list[str],
    rf: RandomForestClassifier,
    transform_meta: dict,
    label: int,
) -> dict | None:
    if df is None or df.empty:
        return None
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        return {"name": name, "error": f"missing {len(miss)} cols", "n": int(len(df))}
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    p = rf.predict_proba(Xt)[:, 1]
    out: dict = {
        "name": name,
        "n": int(len(df)),
        "mean_score": round(float(p.mean()), 4),
        "median_score": round(float(np.median(p)), 4),
    }
    if label == 0:
        out["fpr_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
    else:
        out["recall_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
        for t in (0.15, 0.25, 0.35):
            out[f"recall_at_{t:.2f}_pct"] = round(float((p >= t).mean()) * 100, 2)
    return out


def score_real_distribution(
    rf: RandomForestClassifier,
    feature_cols: list[str],
    transform_meta: dict,
    input_dir: Path,
    max_lines: int,
) -> dict:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        return {"error": f"no jsonl in {input_dir}"}

    rows: list[dict] = []
    bad = 0
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if max_lines > 0 and len(rows) >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                chunk = obj.get("chunk")
                if not isinstance(chunk, list):
                    bad += 1
                    continue
                try:
                    feats = aggregate_chunk_from_miner_payload(chunk)
                except Exception:
                    bad += 1
                    continue
                if not feats:
                    bad += 1
                    continue
                rows.append({
                    "source_file": fp.name,
                    "line_no": line_no,
                    "risk_score_logged": obj.get("risk_score"),
                    **feats,
                })
        if max_lines > 0 and len(rows) >= max_lines:
            break

    if not rows:
        return {"error": "no parsable chunks", "parse_fail": bad}

    df = pd.DataFrame(rows)
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        return {"error": f"missing {len(miss)} feature cols", "parse_fail": bad}

    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    p = rf.predict_proba(Xt)[:, 1]
    out: dict = {
        "n_scored": int(len(df)),
        "parse_fail": bad,
        "model_score_mean": round(float(p.mean()), 4),
        "model_score_median": round(float(np.median(p)), 4),
        "model_bot_pct_ge_0.5": round(float((p >= THRESH).mean()) * 100, 2),
        "model_bot_pct_ge_0.35": round(float((p >= 0.35).mean()) * 100, 2),
        "model_bot_pct_ge_0.20": round(float((p >= 0.20).mean()) * 100, 2),
    }
    rs = df.get("risk_score_logged")
    if rs is not None:
        rs = pd.to_numeric(rs, errors="coerce").fillna(0).values
        out["risk_score_logged_mean"] = round(float(rs.mean()), 4)
        out["risk_score_bot_pct_ge_0.5"] = round(float((rs >= THRESH).mean()) * 100, 2)
        if len(p) > 1 and np.std(p) > 0 and np.std(rs) > 0:
            out["corr_logged_vs_model"] = round(float(np.corrcoef(rs, p)[0, 1]), 4)
    return out


def build_hard_focus_training(
    *,
    gold_train: pd.DataFrame,
    zen: pd.DataFrame | None,
    pub: pd.DataFrame | None,
    fs: pd.DataFrame | None,
    ac: pd.DataFrame | None,
    may8_syn: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    may8_repeat: int,
    may8_cap: int,
    may8_row_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.RandomState(seed)
    parts_h, parts_b = [], []
    weights_h, weights_b = [], []
    src: dict[str, int | float] = {}

    parts_h.append(gold_train[gold_train["label"] == 0][feature_cols])
    weights_h.append(np.ones(len(parts_h[-1])))
    parts_b.append(gold_train[gold_train["label"] == 1][feature_cols])
    weights_b.append(np.ones(len(parts_b[-1])))
    src["gold_human"] = int((gold_train["label"] == 0).sum())
    src["gold_bot"] = int((gold_train["label"] == 1).sum())

    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
        weights_h.append(np.ones(n))
        src["zenodo_human"] = n
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        weights_h.append(np.ones(n))
        src["public_human"] = n
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
        weights_b.append(np.ones(n))
        src["full_spectrum_bot"] = n
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
        weights_b.append(np.ones(n))
        src["acpc_bot"] = n

    n = min(len(may8_syn), may8_cap)
    sub = may8_syn.sample(n=n, random_state=rng)[feature_cols]
    repeated = pd.concat([sub] * max(1, may8_repeat), ignore_index=True)
    parts_b.append(repeated)
    weights_b.append(np.full(len(repeated), float(may8_row_weight)))
    src["may8_reflect_bot_rows"] = n
    src["may8_reflect_bot_effective"] = len(repeated)
    src["may8_row_weight"] = may8_row_weight

    hdf = pd.concat(parts_h, ignore_index=True)
    bdf = pd.concat(parts_b, ignore_index=True)
    X = pd.concat([hdf, bdf], ignore_index=True).values
    y = np.concatenate([np.zeros(len(hdf)), np.ones(len(bdf))])
    sw = np.concatenate([np.concatenate(weights_h), np.concatenate(weights_b)])
    src["total_human"] = int(len(hdf))
    src["total_bot"] = int(len(bdf))
    src["total"] = int(len(X))
    return X, y, sw, src


def may8_hard_easy_split(
    may8_b: pd.DataFrame,
    feature_cols: list[str],
    gold_train: pd.DataFrame,
    zen: pd.DataFrame | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_parts_h = [gold_train[gold_train["label"] == 0][feature_cols]]
    ref_parts_b = [gold_train[gold_train["label"] == 1][feature_cols]]
    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        ref_parts_h.append(zen.sample(n=n, random_state=seed)[feature_cols])
    Xrh = pd.concat(ref_parts_h, ignore_index=True)
    Xrb = pd.concat(ref_parts_b, ignore_index=True)
    Xr = pd.concat([Xrh, Xrb], ignore_index=True).values
    yr = np.concatenate([np.zeros(len(Xrh)), np.ones(len(Xrb))])
    Xrt, ref_tm = tpm.fit_transform_pipeline(Xr, feature_cols)
    ref_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    ref_rf.fit(Xrt, yr)
    ref_p = ref_rf.predict_proba(
        tpm.apply_transform(may8_b[feature_cols].values, feature_cols, ref_tm)
    )[:, 1]
    return may8_b.loc[ref_p < THRESH], may8_b.loc[ref_p >= THRESH]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--may8-synthetic", type=Path, default=DEFAULT_SYN)
    p.add_argument("--may8-repeat", type=int, default=5, help="Repeat synthetic rows in bot pool.")
    p.add_argument("--may8-cap", type=int, default=8000)
    p.add_argument("--may8-row-weight", type=float, default=2.5,
                   help="Sample weight multiplier for each synthetic row.")
    p.add_argument("--bundle-out", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--results-out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    p.add_argument("--real-dist-max-lines", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if not args.may8_synthetic.is_file():
        print(f"[error] missing synthetic parquet: {args.may8_synthetic}")
        return 1

    feature_cols = list(tpm.ROBUST_FEATURES)
    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        print(f"[error] missing {tpm.GOLD_PATH}")
        return 1

    gold_train = gold_full[~gold_full["date"].astype(str).str.contains(MAY8_DATE)]
    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)
    may8_syn = pd.read_parquet(args.may8_synthetic)

    dfs = [gold_train, zen, pub, fs, ac, may8_syn]
    avail = set(feature_cols)
    for df in dfs:
        if df is not None:
            avail &= set(df.columns)
    feature_cols = [c for c in feature_cols if c in avail]
    miss_syn = [c for c in tpm.ROBUST_FEATURES if c not in may8_syn.columns]
    if miss_syn:
        print(f"[error] synthetic missing {len(miss_syn)} features")
        return 1

    print("=" * 70)
    print("MAY-8 HARD-FOCUS RETRAIN + FULL EVAL")
    print("=" * 70)
    print(f"  synthetic: {args.may8_synthetic} ({len(may8_syn)} rows)")
    print(f"  repeat={args.may8_repeat}  cap={args.may8_cap}  row_weight={args.may8_row_weight}")

    X_raw, y, sample_weight, src = build_hard_focus_training(
        gold_train=gold_train,
        zen=zen,
        pub=pub,
        fs=fs,
        ac=ac,
        may8_syn=may8_syn,
        feature_cols=feature_cols,
        seed=args.seed,
        may8_repeat=args.may8_repeat,
        may8_cap=args.may8_cap,
        may8_row_weight=args.may8_row_weight,
    )
    print(f"\n[train] human={src['total_human']} bot={src['total_bot']} total={src['total']}")
    print(f"  sources: {src}")

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_t, y, sample_weight=sample_weight)
    tr_p = rf.predict_proba(X_t)[:, 1]
    print(f"  train AUC={roc_auc_score(y, tr_p):.4f}  "
          f"acc={accuracy_score(y, (tr_p >= THRESH).astype(int)):.4f}")

    may8 = gold_full[gold_full["date"].astype(str).str.contains(MAY8_DATE)]
    may8_b = may8[may8["label"] == 1].copy()
    may8_hard, may8_easy = may8_hard_easy_split(may8_b, feature_cols, gold_train, zen, args.seed)

    test_specs = [
        ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet", 0),
        ("public_test", tpm.TEST_DIR / "public_test_features.parquet", 0),
        ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet", 1),
        ("wsop_stress", tpm.TEST_DIR / "wsop_stress_features.parquet", 0),
    ]

    print("\n" + "=" * 70)
    print("HELD-OUT TEST SETS")
    print("=" * 70)
    test_results: dict = {}
    for name, path, lab in test_specs:
        tdf = load_or_none(path)
        ev = evaluate_block(name, tdf, feature_cols, rf, transform_meta, lab)
        if ev is None:
            print(f"  {name}: SKIP (missing)")
            continue
        test_results[name] = ev
        if lab == 0:
            print(f"  {name:18s} n={ev['n']:5d}  FPR={ev.get('fpr_pct')}%  mean={ev['mean_score']}")
        else:
            print(f"  {name:18s} n={ev['n']:5d}  recall={ev.get('recall_pct')}%  mean={ev['mean_score']}")

    print("\n" + "=" * 70)
    print("MAY-8 HOLD-OUT (not in training)")
    print("=" * 70)
    may8_eval = {
        "may8_human": evaluate_block(
            "may8_human", may8[may8["label"] == 0], feature_cols, rf, transform_meta, 0
        ),
        "may8_bot": evaluate_block("may8_bot", may8_b, feature_cols, rf, transform_meta, 1),
        "may8_hard_bot": evaluate_block(
            "may8_hard_bot", may8_hard, feature_cols, rf, transform_meta, 1
        ),
        "may8_easy_bot": evaluate_block(
            "may8_easy_bot", may8_easy, feature_cols, rf, transform_meta, 1
        ),
    }
    for key in ("may8_human", "may8_bot", "may8_hard_bot", "may8_easy_bot"):
        ev = may8_eval.get(key)
        if not ev:
            continue
        if "fpr" in key or key == "may8_human":
            print(f"  {key:18s} n={ev['n']:4d}  FPR={ev.get('fpr_pct')}%  mean={ev['mean_score']}")
        else:
            print(
                f"  {key:18s} n={ev['n']:4d}  recall@0.5={ev.get('recall_pct')}%  "
                f"@0.25={ev.get('recall_at_0.25_pct')}%  mean={ev['mean_score']}"
            )

    print("\n" + "=" * 70)
    print("REAL DISTRIBUTION (unlabeled)")
    print("=" * 70)
    real_dist = score_real_distribution(
        rf, feature_cols, transform_meta, args.real_dist_dir, args.real_dist_max_lines
    )
    if "error" in real_dist:
        print(f"  [error] {real_dist['error']}")
    else:
        print(f"  n_scored={real_dist['n_scored']}  parse_fail={real_dist.get('parse_fail', 0)}")
        print(f"  model mean={real_dist['model_score_mean']}  median={real_dist['model_score_median']}")
        print(f"  bot% >=0.5: {real_dist['model_bot_pct_ge_0.5']}%  >=0.35: {real_dist['model_bot_pct_ge_0.35']}%")
        if "corr_logged_vs_model" in real_dist:
            print(f"  corr(logged_risk, model)={real_dist['corr_logged_vs_model']}")

    args.bundle_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, args.bundle_out / "lgbm_student.joblib")
    joblib.dump(rf, args.bundle_out / "model.joblib")
    (args.bundle_out / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    (args.bundle_out / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )

    payload = {
        "version": 1,
        "ts": datetime.utcnow().isoformat() + "Z",
        "model": "may8_hard_focus",
        "may8_synthetic": str(args.may8_synthetic),
        "may8_repeat": args.may8_repeat,
        "may8_cap": args.may8_cap,
        "may8_row_weight": args.may8_row_weight,
        "bundle_out": str(args.bundle_out),
        "elapsed_sec": round(time.time() - t0, 1),
        "train_sources": src,
        "test_sets": test_results,
        "may8_holdout": may8_eval,
        "real_distribution": real_dist,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[done] bundle → {args.bundle_out}")
    print(f"[done] results → {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
