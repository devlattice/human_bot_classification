"""Train production model with live-matched targeted bots added.

Reuses the existing pipeline in train_production_model.py but injects the
new live-matched bot parquet as an extra bot source. Saves to a new bundle
directory so v3 / v3c remain untouched, and writes evaluation summary
including May-8 gold + all unseen test sets.

Output bundle:
    workspace/hybrid/model_bundle_v4_targeted/
        lgbm_student.joblib
        feature_cols.json
        transform_meta.json
        retrain_summary.json
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
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score  # noqa: E402

import train_production_model as tpm  # type: ignore  # noqa: E402

TARGETED_BOT_PATH = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "targeted_bot_features.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "workspace" / "hybrid" / "model_bundle_v4_targeted"


def load_datasets_plus(feature_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Existing loader + live-matched bots."""
    datasets = tpm.load_datasets(feature_cols)
    if TARGETED_BOT_PATH.is_file():
        tb = pd.read_parquet(TARGETED_BOT_PATH)
        missing = [f for f in feature_cols if f not in tb.columns]
        if missing:
            print(f"  WARNING: live_matched_bot missing {len(missing)} features: {missing[:5]}")
        else:
            datasets["live_matched_bot"] = tb
            print(f"  Loaded live_matched_bot: {len(tb)} rows  (from {TARGETED_BOT_PATH})")
    else:
        print(f"  SKIP live_matched_bot: {TARGETED_BOT_PATH} not found")
    return datasets


def build_training_with_targeted(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str],
    seed: int,
    targeted_sample: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same as tpm.build_training_data but injects live_matched_bot."""
    rng = np.random.RandomState(seed)
    parts_human, parts_bot = [], []
    meta = {"sources": {}}

    if "gold" in datasets:
        gold = datasets["gold"]
        parts_human.append(gold[gold["label"] == 0][feature_cols])
        parts_bot.append(gold[gold["label"] == 1][feature_cols])
        meta["sources"]["gold_human"] = int((gold["label"] == 0).sum())
        meta["sources"]["gold_bot"] = int((gold["label"] == 1).sum())

    if "zenodo" in datasets:
        zen = datasets["zenodo"]
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_human.append(zen.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["zenodo_human"] = n

    if "public" in datasets:
        pub = datasets["public"]
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_human.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["public_human"] = n

    if "full_spectrum" in datasets:
        fs = datasets["full_spectrum"]
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_bot.append(fs.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["full_spectrum_bot"] = n
    if "gen_bot" in datasets:
        gb = datasets["gen_bot"]
        n = min(len(gb), tpm.BOT_SAMPLE_CAP // 2)
        parts_bot.append(gb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["gen_bot"] = n
    if "cal_bot" in datasets:
        cb = datasets["cal_bot"]
        n = min(len(cb), tpm.BOT_SAMPLE_CAP // 2)
        parts_bot.append(cb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["cal_bot"] = n
    if "acpc_bot" in datasets:
        ac = datasets["acpc_bot"]
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_bot.append(ac.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["acpc_bot"] = n

    if "live_matched_bot" in datasets:
        lmb = datasets["live_matched_bot"]
        n = min(len(lmb), targeted_sample)
        parts_bot.append(lmb.sample(n=n, random_state=rng)[feature_cols])
        meta["sources"]["live_matched_bot"] = n

    human_df = pd.concat(parts_human, ignore_index=True) if parts_human else pd.DataFrame()
    bot_df = pd.concat(parts_bot, ignore_index=True) if parts_bot else pd.DataFrame()
    X = pd.concat([human_df, bot_df], ignore_index=True).values
    y = np.concatenate([np.zeros(len(human_df)), np.ones(len(bot_df))])
    meta["total_human"] = int(len(human_df))
    meta["total_bot"] = int(len(bot_df))
    meta["total"] = int(len(X))
    return X, y, meta


def evaluate_on_test(
    name: str,
    parquet_path: Path,
    feature_cols: list[str],
    rf,
    transform_meta: dict,
    label: int | None,
) -> dict | None:
    if not parquet_path.is_file():
        return None
    df = pd.read_parquet(parquet_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  SKIP {name}: missing {len(missing)} features")
        return None
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    proba = rf.predict_proba(Xt)[:, 1]
    result = {"n": int(len(df)), "mean_score": round(float(proba.mean()), 4)}
    if label == 0:
        result["fpr_pct"] = round(float((proba >= 0.5).mean()) * 100, 3)
        result["human_correct_pct"] = round(float((proba < 0.5).mean()) * 100, 2)
    elif label == 1:
        result["recall_pct"] = round(float((proba >= 0.5).mean()) * 100, 2)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targeted-sample", type=int, default=6000,
                    help="Max chunks to sample from live_matched_bot pool")
    args = ap.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAIN WITH LIVE-MATCHED TARGETED BOTS")
    print("=" * 70)
    print(f"Output: {output_dir}")

    feature_cols = list(tpm.ROBUST_FEATURES)
    print("\nLoading datasets...")
    datasets = load_datasets_plus(feature_cols)

    # Filter feature_cols to those present in all datasets
    avail = set(feature_cols)
    for ds in datasets.values():
        avail &= set(ds.columns)
    feature_cols = [f for f in feature_cols if f in avail]
    print(f"\nUsing {len(feature_cols)} features common to all datasets")

    print("\nBuilding training data (with live-matched bots)...")
    X_raw, y, data_meta = build_training_with_targeted(
        datasets, feature_cols, args.seed, args.targeted_sample
    )
    print(f"  Total: {data_meta['total']} ({data_meta['total_human']} human, {data_meta['total_bot']} bot)")
    print(f"  Source mix: {data_meta['sources']}")

    print("\nFitting transforms (clip + log1p + robust scale)...")
    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    print(f"  clip={transform_meta['clip_features_count']}  "
          f"log1p={transform_meta['log1p_selected_count']}  "
          f"scale={transform_meta['robust_scale_features_count']}")

    print("\nTraining RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=args.seed, n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_t, y)
    print(f"  trained in {time.time() - t0:.1f}s")

    train_proba = rf.predict_proba(X_t)[:, 1]
    train_auc = roc_auc_score(y, train_proba)
    train_acc = accuracy_score(y, (train_proba >= 0.5).astype(int))
    print(f"  Train AUC={train_auc:.4f}  Acc={train_acc:.4f}")

    # ─── Per-day gold evaluation (in-sample but stratified) ───
    print("\n" + "=" * 70)
    print("GOLD PER-DAY EVALUATION (held-out: each day applied as test)")
    print("=" * 70)
    gold_results = []
    if "gold" in datasets:
        gold = datasets["gold"]
        for date in sorted(gold["date"].unique()):
            sub = gold[gold["date"] == date]
            Xs = tpm.apply_transform(sub[feature_cols].values, feature_cols, transform_meta)
            proba = rf.predict_proba(Xs)[:, 1]
            ys = sub["label"].values
            humans = proba[ys == 0]
            bots = proba[ys == 1]
            res = {
                "date": date,
                "n": int(len(sub)),
                "human_n": int((ys == 0).sum()),
                "bot_n": int((ys == 1).sum()),
                "human_fpr_pct": round(float((humans >= 0.5).mean()) * 100, 3) if len(humans) else None,
                "bot_recall_pct": round(float((bots >= 0.5).mean()) * 100, 2) if len(bots) else None,
                "bot_mean_score": round(float(bots.mean()), 4) if len(bots) else None,
                "human_mean_score": round(float(humans.mean()), 4) if len(humans) else None,
            }
            gold_results.append(res)
            marker = " <-- May-8" if "05-08" in date else ""
            print(f"  {date} (n={res['n']:4d})  FPR={res['human_fpr_pct']!s:>6}%  "
                  f"BotRecall={res['bot_recall_pct']!s:>5}%  "
                  f"bot_mean={res['bot_mean_score']!s:>7}  "
                  f"hum_mean={res['human_mean_score']!s:>7}{marker}")

    # ─── Unseen test sets ───
    print("\n" + "=" * 70)
    print("UNSEEN TEST SETS")
    print("=" * 70)
    test_specs = [
        ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet", 0),
        ("public_test", tpm.TEST_DIR / "public_test_features.parquet", 0),
        ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet", 1),
    ]
    # Optional WSOP stress (human)
    wsop = tpm.TEST_DIR / "wsop_stress_features.parquet"
    if wsop.is_file():
        test_specs.append(("wsop_human_stress", wsop, 0))

    test_results = {}
    for name, path, label in test_specs:
        r = evaluate_on_test(name, path, feature_cols, rf, transform_meta, label)
        if r is None:
            continue
        test_results[name] = r
        if label == 0:
            print(f"  {name:22s} n={r['n']:5d}  FPR={r['fpr_pct']:.3f}%  "
                  f"correct={r['human_correct_pct']:.1f}%  mean={r['mean_score']:.4f}")
        else:
            print(f"  {name:22s} n={r['n']:5d}  recall={r['recall_pct']:.1f}%  "
                  f"mean={r['mean_score']:.4f}")

    # ─── Test on logged unlabeled (live distribution sanity) ───
    print("\n" + "=" * 70)
    print("LIVE UNLABELED (from real_distribution) — should be bimodal now")
    print("=" * 70)
    unlabeled_parquet = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "unlabeled_features.parquet"
    if unlabeled_parquet.is_file():
        u = pd.read_parquet(unlabeled_parquet)
        missing = [c for c in feature_cols if c not in u.columns]
        if not missing:
            uX = tpm.apply_transform(u[feature_cols].values, feature_cols, transform_meta)
            up = rf.predict_proba(uX)[:, 1]
            print(f"  n={len(u)}  mean={up.mean():.4f}  median={np.median(up):.4f}  "
                  f">=0.5: {int((up >= 0.5).sum())}/{len(u)} ({100*(up >= 0.5).mean():.1f}%)")
            for q in (10, 25, 50, 75, 90, 95):
                print(f"    p{q:02d}={np.percentile(up, q):.4f}")
        else:
            print(f"  SKIP: live unlabeled missing {len(missing)} features")

    # ─── Save bundle ───
    import joblib

    joblib.dump(rf, output_dir / "lgbm_student.joblib")
    (output_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    (output_dir / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )
    summary = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model": "v4 with live-matched targeted bots",
        "n_features": len(feature_cols),
        "n_training_samples": int(len(X_raw)),
        "data_meta": data_meta,
        "train_auc": round(float(train_auc), 6),
        "gold_per_day": gold_results,
        "unseen_test_results": test_results,
    }
    (output_dir / "retrain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[done] bundle saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
