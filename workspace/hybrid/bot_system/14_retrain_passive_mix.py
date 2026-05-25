"""Retrain production RF with passive synthetic bots (2×) mixed into bot pool.

Expects ``passive_matched_bot_features.parquet`` from passive LHS + generation.
Evaluates the same surfaces as ``train_production_model.py`` plus per-day gold.

Usage:
    python workspace/hybrid/bot_system/14_retrain_passive_mix.py \\
        --passive-parquet workspace/hybrid/bot_system/data/passive_matched_bot_features.parquet
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

import joblib  # noqa: E402
import train_production_model as tpm  # type: ignore  # noqa: E402

DEFAULT_PASSIVE = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "passive_matched_bot_features.parquet"
DEFAULT_LIVE = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "targeted_bot_features.parquet"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "model_bundle_v5_passive"


def build_training_with_passive(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str],
    seed: int,
    passive_repeat: int,
    passive_cap: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.RandomState(seed)
    parts_h, parts_b = [], []
    meta: dict = {"sources": {}}

    if "gold" in datasets:
        gold = datasets["gold"]
        parts_h.append(gold[gold["label"] == 0][feature_cols])
        parts_b.append(gold[gold["label"] == 1][feature_cols])
        meta["sources"]["gold_human"] = int((gold["label"] == 0).sum())
        meta["sources"]["gold_bot"] = int((gold["label"] == 1).sum())

    if "zenodo" in datasets:
        zen = datasets["zenodo"]
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["zenodo_human"] = n
    if "public" in datasets:
        pub = datasets["public"]
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["public_human"] = n
    if "full_spectrum" in datasets:
        fs = datasets["full_spectrum"]
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["full_spectrum_bot"] = n
    if "gen_bot" in datasets:
        gb = datasets["gen_bot"]
        n = min(len(gb), tpm.BOT_SAMPLE_CAP // 2)
        parts_b.append(gb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["gen_bot"] = n
    if "cal_bot" in datasets:
        cb = datasets["cal_bot"]
        n = min(len(cb), tpm.BOT_SAMPLE_CAP // 2)
        parts_b.append(cb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["cal_bot"] = n
    if "acpc_bot" in datasets:
        ac = datasets["acpc_bot"]
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["acpc_bot"] = n
    if "live_matched_bot" in datasets:
        lm = datasets["live_matched_bot"]
        n = min(len(lm), 6000)
        parts_b.append(lm.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["live_matched_bot"] = n

    if "passive_matched_bot" in datasets:
        pb = datasets["passive_matched_bot"]
        n = min(len(pb), passive_cap)
        sub = pb.sample(n=n, random_state=rng)[feature_cols]
        repeated = pd.concat([sub] * max(1, passive_repeat), ignore_index=True)
        parts_b.append(repeated)
        meta["sources"]["passive_matched_bot_rows"] = n
        meta["sources"]["passive_matched_bot_effective"] = len(repeated)

    hdf = pd.concat(parts_h, ignore_index=True) if parts_h else pd.DataFrame()
    bdf = pd.concat(parts_b, ignore_index=True) if parts_b else pd.DataFrame()
    X = pd.concat([hdf, bdf], ignore_index=True).values
    y = np.concatenate([np.zeros(len(hdf)), np.ones(len(bdf))])
    meta["total_human"] = int(len(hdf))
    meta["total_bot"] = int(len(bdf))
    meta["total"] = int(len(X))
    return X, y, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--passive-parquet", type=Path, default=DEFAULT_PASSIVE)
    p.add_argument("--no-live-matched", action="store_true",
                   help="Do not mix targeted_bot_features.parquet into training.")
    p.add_argument("--passive-repeat", type=int, default=2)
    p.add_argument("--passive-cap", type=int, default=8000)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rf-params-json",
        type=Path,
        default=None,
        help="Same as train_production_model.py: Optuna JSON overrides RF CLI flags.",
    )
    tpm.add_rf_arguments(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PASSIVE MIX RETRAIN")
    print("=" * 70)
    print(f"Output: {out}")

    feature_cols = list(tpm.ROBUST_FEATURES)
    datasets = tpm.load_datasets(feature_cols)

    if args.passive_parquet.is_file():
        pbf = pd.read_parquet(args.passive_parquet)
        miss = [c for c in feature_cols if c not in pbf.columns]
        if miss:
            print(f"[warn] passive parquet missing {len(miss)} features; skipping passive")
        else:
            datasets["passive_matched_bot"] = pbf
            print(f"  Loaded passive_matched_bot: {len(pbf)} rows")
    else:
        print(f"[warn] no passive parquet at {args.passive_parquet}")

    live_path = DEFAULT_LIVE
    if not args.no_live_matched and live_path.is_file():
        lm = pd.read_parquet(live_path)
        miss = [c for c in feature_cols if c not in lm.columns]
        if not miss:
            datasets["live_matched_bot"] = lm
            print(f"  Loaded live_matched_bot: {len(lm)} rows")

    avail = set(feature_cols)
    for ds in datasets.values():
        avail &= set(ds.columns)
    feature_cols = [f for f in feature_cols if f in avail]
    print(f"\nUsing {len(feature_cols)} features")

    X_raw, y, data_meta = build_training_with_passive(
        datasets, feature_cols, args.seed, args.passive_repeat, args.passive_cap
    )
    print(f"\nTraining mix: {data_meta['total']} ({data_meta['total_human']} human, "
          f"{data_meta['total_bot']} bot)")
    print(f"  sources: {data_meta['sources']}")

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf_kwargs = tpm.rf_kwargs_from_namespace(args, args.seed)
    if args.rf_params_json is not None and args.rf_params_json.is_file():
        rf_kwargs = tpm.apply_rf_params_patch(
            rf_kwargs, tpm.load_rf_params_json(args.rf_params_json)
        )
        print(f"\n  RF patch from {args.rf_params_json}")
    print("\nTraining RandomForest…")
    print(f"  RF params: {tpm.rf_params_for_summary(rf_kwargs)}")
    t0 = time.time()
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X_t, y)
    print(f"  done in {time.time() - t0:.1f}s")

    tr_p = rf.predict_proba(X_t)[:, 1]
    print(f"  train AUC={roc_auc_score(y, tr_p):.4f}  "
          f"acc={accuracy_score(y, (tr_p >= 0.5).astype(int)):.4f}")

    # LOOCV gold
    cv_results = []
    mean_auc = min_auc = None
    if "gold" in datasets:
        print("\nLeave-one-day-out on gold…")
        ext_h, ext_b = [], []
        rng = np.random.RandomState(args.seed)
        if "zenodo" in datasets:
            ext_h.append(datasets["zenodo"].sample(
                n=min(len(datasets["zenodo"]), tpm.HUMAN_SAMPLE_CAP), random_state=rng
            )[feature_cols].values)
        if "public" in datasets:
            pub_up = pd.concat([datasets["public"]] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
            ext_h.append(pub_up.sample(
                n=min(len(pub_up), tpm.HUMAN_SAMPLE_CAP), random_state=rng
            )[feature_cols].values)
        if "full_spectrum" in datasets:
            ext_b.append(datasets["full_spectrum"].sample(
                n=min(len(datasets["full_spectrum"]), tpm.BOT_SAMPLE_CAP), random_state=rng
            )[feature_cols].values)
        if "gen_bot" in datasets:
            ext_b.append(datasets["gen_bot"].sample(
                n=min(len(datasets["gen_bot"]), tpm.BOT_SAMPLE_CAP // 2), random_state=rng
            )[feature_cols].values)
        if "cal_bot" in datasets:
            ext_b.append(datasets["cal_bot"].sample(
                n=min(len(datasets["cal_bot"]), tpm.BOT_SAMPLE_CAP // 2), random_state=rng
            )[feature_cols].values)
        if "acpc_bot" in datasets:
            ext_b.append(datasets["acpc_bot"].sample(
                n=min(len(datasets["acpc_bot"]), tpm.BOT_SAMPLE_CAP), random_state=rng
            )[feature_cols].values)
        if "live_matched_bot" in datasets:
            lm = datasets["live_matched_bot"]
            ext_b.append(lm.sample(n=min(len(lm), 6000), random_state=rng)[feature_cols].values)
        if "passive_matched_bot" in datasets:
            pm = datasets["passive_matched_bot"]
            ext_b.append(pm.sample(n=min(len(pm), args.passive_cap), random_state=rng)[feature_cols].values)

        ext_human = np.vstack(ext_h) if ext_h else np.empty((0, len(feature_cols)))
        ext_bot = np.vstack(ext_b) if ext_b else np.empty((0, len(feature_cols)))
        cv_results = tpm.evaluate_gold_loocv(
            datasets["gold"], ext_human, ext_bot, feature_cols, transform_meta, rf_kwargs
        )
        for r in cv_results:
            m = " <-- May-8" if "05-08" in str(r["date"]) else ""
            print(f"  {r['date']}: AUC={r['auc']:.4f}  acc={r['accuracy']:.4f}  "
                  f"bot_detect={r['bot_detect_rate']:.3f}{m}")
        aucs = [r["auc"] for r in cv_results if not np.isnan(r["auc"])]
        mean_auc = float(np.mean(aucs)) if aucs else None
        min_auc = float(np.min(aucs)) if aucs else None
        print(f"  mean AUC={mean_auc:.4f}  min AUC={min_auc:.4f}" if mean_auc is not None else "  mean AUC=n/a")

    # Unseen + WSOP
    test_results: dict = {}
    specs = [
        ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet", 0),
        ("public_test", tpm.TEST_DIR / "public_test_features.parquet", 0),
        ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet", 1),
    ]
    wsop = tpm.TEST_DIR / "wsop_stress_features.parquet"
    if wsop.is_file():
        specs.append(("wsop_stress", wsop, 0))

    print("\n" + "=" * 70)
    print("UNSEEN / STRESS TESTS")
    print("=" * 70)
    for name, path, lab in specs:
        if not path.is_file():
            continue
        tdf = pd.read_parquet(path)
        if any(c not in tdf.columns for c in feature_cols):
            print(f"  SKIP {name}: missing features")
            continue
        tX = tpm.apply_transform(tdf[feature_cols].values, feature_cols, transform_meta)
        pr = rf.predict_proba(tX)[:, 1]
        if lab == 0:
            fpr = float((pr >= 0.5).mean()) * 100
            test_results[name] = {
                "n": len(tdf), "fpr_pct": round(fpr, 3),
                "mean_score": round(float(pr.mean()), 4),
            }
            print(f"  {name:18s} n={len(tdf):5d}  FPR={fpr:.3f}%  mean={pr.mean():.4f}")
        else:
            rec = float((pr >= 0.5).mean()) * 100
            test_results[name] = {
                "n": len(tdf), "recall_pct": round(rec, 2),
                "mean_score": round(float(pr.mean()), 4),
            }
            print(f"  {name:18s} n={len(tdf):5d}  recall={rec:.1f}%  mean={pr.mean():.4f}")

    # Gold per-day (in-sample diagnostic)
    gold_per_day = []
    if "gold" in datasets:
        gold = datasets["gold"]
        print("\n" + "=" * 70)
        print("GOLD PER-DAY (in-sample)")
        print("=" * 70)
        for date in sorted(gold["date"].unique()):
            sub = gold[gold["date"] == date]
            Xs = tpm.apply_transform(sub[feature_cols].values, feature_cols, transform_meta)
            pr = rf.predict_proba(Xs)[:, 1]
            ys = sub["label"].values
            h = pr[ys == 0]
            b = pr[ys == 1]
            rec = float((b >= 0.5).mean()) * 100 if len(b) else None
            fpr = float((h >= 0.5).mean()) * 100 if len(h) else None
            gold_per_day.append({
                "date": date, "n": int(len(sub)),
                "human_fpr_pct": round(fpr, 3) if fpr is not None else None,
                "bot_recall_pct": round(rec, 2) if rec is not None else None,
                "bot_mean": round(float(b.mean()), 4) if len(b) else None,
            })
            m = " <-- May-8" if "05-08" in str(date) else ""
            print(f"  {date}  FPR={fpr:.3f}%  bot_recall={rec:.1f}%  bot_mean={b.mean():.4f}{m}")

    # May-8 detail
    may8_block = {}
    if "gold" in datasets:
        g = datasets["gold"]
        m8 = g[g["date"].str.contains("05-08")]
        if len(m8):
            X8 = tpm.apply_transform(m8[feature_cols].values, feature_cols, transform_meta)
            p8 = rf.predict_proba(X8)[:, 1]
            mh = p8[m8["label"].values == 0]
            mb = p8[m8["label"].values == 1]
            may8_block = {
                "human_mean": round(float(mh.mean()), 4) if len(mh) else None,
                "bot_mean": round(float(mb.mean()), 4) if len(mb) else None,
                "bot_recall_0.5_pct": round(float((mb >= 0.5).mean()) * 100, 2) if len(mb) else None,
            }
            print(f"\nMay-8 detail: human_mean={may8_block['human_mean']}  "
                  f"bot_mean={may8_block['bot_mean']}  bot_recall@0.5={may8_block['bot_recall_0.5_pct']}%")

    joblib.dump(rf, out / "lgbm_student.joblib")
    (out / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    (out / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )
    summary = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model": "passive_mix_retrain",
        "passive_parquet": str(args.passive_parquet),
        "passive_repeat": args.passive_repeat,
        "passive_cap": args.passive_cap,
        "sources": data_meta["sources"],
        "n_training": int(len(X_raw)),
        "model_type": "RandomForestClassifier",
        "model_params": tpm.rf_params_for_summary(rf_kwargs),
        "gold_loocv": cv_results,
        "gold_loocv_mean_auc": mean_auc,
        "gold_loocv_min_auc": min_auc,
        "unseen_tests": test_results,
        "gold_per_day": gold_per_day,
        "may8": may8_block,
    }
    (out / "retrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] bundle → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
