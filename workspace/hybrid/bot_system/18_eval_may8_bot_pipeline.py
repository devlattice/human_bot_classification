"""Train with May-8-reflect synthetic bots; evaluate test + May-8 + real_distribution.

Trains RandomForest on gold (May-8 excluded) + standard corpora + ``--may8-matched``
parquet, then reports held-out metrics and scores unlabeled validator logs.

Output:
  workspace/hybrid/bot_system/data/may8_bot_pipeline_results.json

Usage:
  python workspace/hybrid/bot_system/18_eval_may8_bot_pipeline.py \\
    --may8-matched workspace/hybrid/bot_system/data/may8_reflect_bot_features.parquet
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
DEFAULT_MATCHED = DATA / "may8_reflect_bot_features.parquet"
DEFAULT_OUT = DATA / "may8_bot_pipeline_results.json"
DEFAULT_BUNDLE = REPO / "workspace" / "hybrid" / "model_bundle_may8_reflect"
REAL_DIST = REPO / "workspace" / "dataset" / "real_distribution"
THRESH = 0.5


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
                    raw = aggregate_chunk_from_miner_payload(chunk)
                except Exception:
                    bad += 1
                    continue
                if not raw or any(c not in raw for c in feature_cols):
                    bad += 1
                    continue
                Xv = tpm.apply_transform(
                    np.asarray([raw[c] for c in feature_cols], dtype=np.float64)[None, :],
                    feature_cols,
                    transform_meta,
                )
                score = float(rf.predict_proba(Xv)[0, 1])
                rs = obj.get("risk_score")
                rows.append({
                    "risk_score_logged": float(rs) if isinstance(rs, (int, float)) else None,
                    "model_score": score,
                    "pred_bot": score >= THRESH,
                })
        if max_lines > 0 and len(rows) >= max_lines:
            break

    if not rows:
        return {"error": "no scored rows", "parse_fail": bad}

    df = pd.DataFrame(rows)
    ms = df["model_score"]
    rs = df["risk_score_logged"].dropna()
    out: dict = {
        "n_scored": int(len(df)),
        "parse_fail": int(bad),
        "model_score_mean": round(float(ms.mean()), 4),
        "model_score_median": round(float(ms.median()), 4),
        "model_bot_pct_ge_0.5": round(float((ms >= THRESH).mean()) * 100, 2),
        "model_bot_pct_ge_0.35": round(float((ms >= 0.35).mean()) * 100, 2),
        "model_bot_pct_ge_0.20": round(float((ms >= 0.20).mean()) * 100, 2),
    }
    if len(rs) >= 10:
        sub = df.dropna(subset=["risk_score_logged"])
        out["risk_score_logged_mean"] = round(float(rs.mean()), 4)
        out["risk_score_bot_pct_ge_0.5"] = round(float((rs >= THRESH).mean()) * 100, 2)
        out["corr_logged_vs_model"] = round(
            float(np.corrcoef(sub["risk_score_logged"].values, sub["model_score"].values)[0, 1]),
            4,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--may8-matched", type=Path, default=DEFAULT_MATCHED)
    p.add_argument("--may8-bot-cap", type=int, default=12000)
    p.add_argument("--bundle-out", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--results-out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    p.add_argument("--real-dist-max-lines", type=int, default=0, help="0 = all lines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-may8-recall", type=float, default=80.0)
    p.add_argument("--max-human-fpr", type=float, default=2.0)
    p.add_argument("--max-zenodo-fpr", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    if not args.may8_matched.is_file():
        print(f"[error] missing {args.may8_matched}")
        return 1

    feature_cols = list(tpm.ROBUST_FEATURES)
    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        print(f"[error] missing {tpm.GOLD_PATH}")
        return 1

    gold_train = gold_full[~gold_full["date"].astype(str).str.contains("2026-05-08")]
    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)
    mmb = load_or_none(args.may8_matched)

    dfs = [gold_train, zen, pub, fs, ac, mmb]
    avail = set(feature_cols)
    for df in dfs:
        if df is not None:
            avail &= set(df.columns)
    feature_cols = [c for c in feature_cols if c in avail]

    rng = np.random.RandomState(args.seed)
    parts_h, parts_b = [], []
    src: dict[str, int] = {}

    parts_h.append(gold_train[gold_train["label"] == 0][feature_cols])
    parts_b.append(gold_train[gold_train["label"] == 1][feature_cols])
    src["gold_human"] = int((gold_train["label"] == 0).sum())
    src["gold_bot"] = int((gold_train["label"] == 1).sum())

    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
        src["zenodo_human"] = n
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        src["public_human"] = n
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
        src["full_spectrum_bot"] = n
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
        src["acpc_bot"] = n
    if mmb is not None:
        n = min(len(mmb), args.may8_bot_cap)
        parts_b.append(mmb.sample(n=n, random_state=rng)[feature_cols])
        src["may8_reflect_bot"] = n

    Xh = pd.concat(parts_h, ignore_index=True)
    Xb = pd.concat(parts_b, ignore_index=True)
    X_raw = pd.concat([Xh, Xb], ignore_index=True).values
    y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])
    print(f"[train] human={len(Xh)} bot={len(Xb)} features={len(feature_cols)} sources={src}")

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_t, y)
    train_p = rf.predict_proba(X_t)[:, 1]
    train_auc = float(roc_auc_score(y, train_p))
    train_acc = float(accuracy_score(y, (train_p >= THRESH).astype(int)))

    # Fixed hard/easy split (reference hold-out on May-8 bots)
    may8 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-08")]
    may8_b = may8[may8["label"] == 1].copy()
    ref_train = gold_full[~gold_full["date"].astype(str).str.contains("2026-05-08")]
    ref_parts_h = [ref_train[ref_train["label"] == 0][feature_cols]]
    ref_parts_b = [ref_train[ref_train["label"] == 1][feature_cols]]
    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        ref_parts_h.append(zen.sample(n=n, random_state=args.seed)[feature_cols])
    Xrh = pd.concat(ref_parts_h, ignore_index=True)
    Xrb = pd.concat(ref_parts_b, ignore_index=True)
    Xr = pd.concat([Xrh, Xrb], ignore_index=True).values
    yr = np.concatenate([np.zeros(len(Xrh)), np.ones(len(Xrb))])
    Xrt, ref_tm = tpm.fit_transform_pipeline(Xr, feature_cols)
    ref_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=args.seed, n_jobs=-1, class_weight="balanced",
    )
    ref_rf.fit(Xrt, yr)
    ref_p = ref_rf.predict_proba(
        tpm.apply_transform(may8_b[feature_cols].values, feature_cols, ref_tm)
    )[:, 1]
    may8_hard = may8_b.loc[ref_p < THRESH]
    may8_easy = may8_b.loc[ref_p >= THRESH]

    may7 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-07")]
    evaluations = {
        "may8_human": evaluate_block(
            "may8_human", may8[may8["label"] == 0], feature_cols, rf, transform_meta, 0
        ),
        "may8_bot": evaluate_block(
            "may8_bot", may8_b, feature_cols, rf, transform_meta, 1
        ),
        "may8_hard_bot": evaluate_block(
            "may8_hard_bot", may8_hard, feature_cols, rf, transform_meta, 1
        ),
        "may8_easy_bot": evaluate_block(
            "may8_easy_bot", may8_easy, feature_cols, rf, transform_meta, 1
        ),
        "may7_human": evaluate_block(
            "may7_human", may7[may7["label"] == 0], feature_cols, rf, transform_meta, 0
        ),
        "may7_bot": evaluate_block(
            "may7_bot", may7[may7["label"] == 1], feature_cols, rf, transform_meta, 1
        ),
        "zenodo_test": evaluate_block(
            "zenodo_test",
            load_or_none(tpm.TEST_DIR / "zenodo_test_features.parquet"),
            feature_cols, rf, transform_meta, 0,
        ),
        "public_test": evaluate_block(
            "public_test",
            load_or_none(tpm.TEST_DIR / "public_test_features.parquet"),
            feature_cols, rf, transform_meta, 0,
        ),
        "acpc_bot_test": evaluate_block(
            "acpc_bot_test",
            load_or_none(tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
            feature_cols, rf, transform_meta, 1,
        ),
    }

    real_dist = score_real_distribution(
        rf, feature_cols, transform_meta, args.real_dist_dir, args.real_dist_max_lines
    )

    args.bundle_out.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, args.bundle_out / "model.joblib")
    joblib.dump(rf, args.bundle_out / "lgbm_student.joblib")
    (args.bundle_out / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    (args.bundle_out / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )

    may8_rec = (evaluations.get("may8_bot") or {}).get("recall_pct")
    may8_fpr = (evaluations.get("may8_human") or {}).get("fpr_pct")
    zen_fpr = (evaluations.get("zenodo_test") or {}).get("fpr_pct")

    payload = {
        "version": 1,
        "ts": datetime.utcnow().isoformat() + "Z",
        "may8_matched_parquet": str(args.may8_matched),
        "bundle_out": str(args.bundle_out),
        "elapsed_sec": round(time.time() - t0, 1),
        "train": {
            "train_auc": round(train_auc, 4),
            "train_acc": round(train_acc, 4),
            "sources": src,
            "n_features": len(feature_cols),
        },
        "evaluations": evaluations,
        "real_distribution": real_dist,
        "gates": {
            "may8_recall_pct": may8_rec,
            "may8_human_fpr_pct": may8_fpr,
            "zenodo_fpr_pct": zen_fpr,
            "pass_may8_recall": may8_rec is not None and may8_rec >= args.min_may8_recall,
            "pass_may8_fpr": may8_fpr is not None and may8_fpr <= args.max_human_fpr,
            "pass_zen_fpr": zen_fpr is None or zen_fpr <= args.max_zenodo_fpr,
        },
    }
    payload["gates"]["overall_pass"] = all(
        payload["gates"][k] for k in ("pass_may8_recall", "pass_may8_fpr", "pass_zen_fpr")
    )

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("EVALUATION (May-8 reflect pipeline)")
    print("=" * 70)
    for key in (
        "may8_bot", "may8_hard_bot", "may8_easy_bot", "may8_human",
        "zenodo_test", "public_test", "acpc_bot_test",
    ):
        b = evaluations.get(key)
        if not b or "error" in b:
            continue
        if "recall_pct" in b:
            print(f"  {key:<18} recall={b['recall_pct']}%  n={b['n']}")
        else:
            print(f"  {key:<18} FPR={b['fpr_pct']}%  n={b['n']}")
    print(f"\n  real_distribution: {json.dumps(real_dist, indent=2)}")
    print(f"\n  gates: {payload['gates']}")
    print(f"\n[done] {args.results_out}")
    print(f"       bundle: {args.bundle_out}")
    return 0 if payload["gates"]["overall_pass"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
