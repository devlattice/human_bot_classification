"""Train v11 production model: full train + 80% May-8 (×3 repeat), eval 20% May-8 + real_distribution.

Usage:
  python workspace/hybrid/scripts/train_v11_may8_blend.py \\
    --output-dir workspace/model/artifacts/model_bundle_v11_prod

Or: ./workspace/model/deploy_v11_prod.sh
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
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_production_model as tpm  # noqa: E402
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402

REAL_DIST = REPO / "workspace/dataset/real_distribution"
FEATURES_JSON = REPO / "workspace/hybrid/selected_features_v3.json"
MAY8_PATH = tpm.MAY8_GOLD_TEST_PATH
HOLDOUT_PATH = tpm.TEST_DIR / "may8_holdout_20pct.parquet"
DEFAULT_OUT = REPO / "workspace/model/artifacts/model_bundle_v11_prod"
EVAL_JSON = REPO / "workspace/hybrid/bot_system/data/v11_prod_eval.json"


def load_feature_cols(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["selected_features"])


def build_training_with_may8(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str],
    may8_train: pd.DataFrame,
    *,
    may8_repeat: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    X, y, meta = tpm.build_training_data(datasets, feature_cols, seed)
    m8 = may8_train
    n_base = len(m8)
    parts = [m8] * max(1, may8_repeat)
    may8_up = pd.concat(parts, ignore_index=True)
    X_m = may8_up[feature_cols].values
    y_m = may8_up["label"].values.astype(int)
    X_out = np.vstack([X, X_m])
    y_out = np.concatenate([y, y_m])
    meta["sources"]["may8_train_base"] = int(n_base)
    meta["sources"]["may8_train_effective"] = int(len(may8_up))
    meta["may8_repeat"] = int(may8_repeat)
    meta["total"] = int(len(X_out))
    meta["total_bot"] = int((y_out == 1).sum())
    meta["total_human"] = int((y_out == 0).sum())
    return X_out, y_out, meta


def eval_labeled(
    name: str,
    df: pd.DataFrame,
    rf: RandomForestClassifier,
    feature_cols: list[str],
    transform_meta: dict,
    thresh: float = 0.5,
) -> dict:
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        return {"error": f"missing {len(miss)} features"}
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    p = rf.predict_proba(Xt)[:, 1]
    y = df["label"].values.astype(int)
    out: dict = {
        "n": int(len(df)),
        "mean_score": round(float(p.mean()), 4),
        "pct_scores_0.5_1.0": round(float(((p >= 0.5) & (p <= 1.0)).mean()) * 100, 2),
    }
    if (y == 0).any():
        out["human_fpr_pct"] = round(float((p[y == 0] >= thresh).mean()) * 100, 3)
    if (y == 1).any():
        out["bot_recall_pct"] = round(float((p[y == 1] >= thresh).mean()) * 100, 2)
        out["bot_mean_score"] = round(float(p[y == 1].mean()), 4)
    return out


def eval_real_distribution(
    rf: RandomForestClassifier,
    feature_cols: list[str],
    transform_meta: dict,
    real_dir: Path,
    thresh: float = 0.5,
) -> dict:
    logged: list[float] = []
    model: list[float] = []
    bad = 0
    for fp in sorted(real_dir.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunk = obj.get("chunk")
                    rs = obj.get("risk_score")
                    if not isinstance(chunk, list):
                        bad += 1
                        continue
                    raw = aggregate_chunk_from_miner_payload(chunk)
                    if not raw or any(c not in raw for c in feature_cols):
                        bad += 1
                        continue
                    Xv = tpm.apply_transform(
                        np.asarray([raw[c] for c in feature_cols], dtype=np.float64)[None, :],
                        feature_cols,
                        transform_meta,
                    )
                    p = float(rf.predict_proba(Xv)[0, 1])
                    model.append(p)
                    if isinstance(rs, (int, float)):
                        logged.append(float(rs))
                except Exception:
                    bad += 1
    if not model:
        return {"error": "no scores"}
    m = np.asarray(model)
    out = {
        "n": int(len(m)),
        "parse_fail": bad,
        "mean_score": round(float(m.mean()), 4),
        "median_score": round(float(np.median(m)), 4),
        "pct_scores_0.5_1.0": round(float(((m >= 0.5) & (m <= 1.0)).mean()) * 100, 2),
        "pct_pred_bot_0.5": round(float((m >= thresh).mean()) * 100, 2),
    }
    if len(logged) >= 10:
        lg = np.asarray(logged[: len(m)])
        out["logged_risk_mean"] = round(float(lg.mean()), 4)
        out["logged_pct_ge_0.5"] = round(float((lg >= thresh).mean()) * 100, 2)
        out["corr_vs_logged_risk"] = round(float(np.corrcoef(lg, m[: len(lg)])[0, 1]), 4)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--features-json", type=Path, default=FEATURES_JSON)
    p.add_argument("--may8-frac-train", type=float, default=0.8)
    p.add_argument("--may8-repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rf-params-json",
        type=Path,
        default=REPO / "workspace/hybrid/model_bundle/best_rf_params.json",
    )
    p.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    p.add_argument("--eval-json", type=Path, default=EVAL_JSON)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tpm.TEST_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_cols(args.features_json)
    print("=" * 70)
    print("V11 PRODUCTION TRAIN — May-8 80% blend × repeat")
    print("=" * 70)
    print(f"Features: {len(feature_cols)}")
    print(f"Output: {args.output_dir}")

    may8 = pd.read_parquet(MAY8_PATH)
    may8_train, may8_holdout = train_test_split(
        may8,
        train_size=args.may8_frac_train,
        random_state=args.seed,
        stratify=may8["label"],
    )
    may8_holdout.to_parquet(HOLDOUT_PATH, index=False)
    print(f"\nMay-8 split: train={len(may8_train)} ({args.may8_frac_train:.0%})  "
          f"holdout={len(may8_holdout)} ({1-args.may8_frac_train:.0%})")
    print(f"  holdout saved: {HOLDOUT_PATH}")
    print(f"  train effective rows: {len(may8_train) * args.may8_repeat} (×{args.may8_repeat})")

    datasets = tpm.load_datasets(feature_cols)
    X_raw, y, data_meta = build_training_with_may8(
        datasets,
        feature_cols,
        may8_train,
        may8_repeat=args.may8_repeat,
        seed=args.seed,
    )
    print(f"\nTraining mix: {data_meta['total']} rows "
          f"({data_meta['total_human']} human, {data_meta['total_bot']} bot)")
    src = data_meta.get("sources", {})
    print(f"  may8 in mix: {src.get('may8_train_effective')} "
          f"(base {src.get('may8_train_base')}, ×{data_meta.get('may8_repeat')})")

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf_kwargs = tpm.rf_kwargs_from_namespace(
        argparse.Namespace(
            rf_n_estimators=300,
            rf_max_depth="6",
            rf_min_samples_leaf=15,
            rf_min_samples_split=2,
            rf_max_features="sqrt",
            rf_max_samples=1.0,
            rf_class_weight="balanced",
            rf_ccp_alpha=0.0,
        ),
        args.seed,
    )
    if args.rf_params_json.is_file():
        rf_kwargs = tpm.apply_rf_params_patch(rf_kwargs, tpm.load_rf_params_json(args.rf_params_json))

    print("\nTraining RF...")
    t0 = time.time()
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X_t, y)
    print(f"  done in {time.time() - t0:.1f}s")

    train_p = rf.predict_proba(X_t)[:, 1]
    print(f"  train AUC={roc_auc_score(y, train_p):.4f}  acc={accuracy_score(y, train_p>=0.5):.4f}")

    optimal_t = tpm.find_optimal_threshold(y, train_p)
    joblib.dump(rf, args.output_dir / "model.joblib")
    (args.output_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8"
    )
    (args.output_dir / "transform_meta.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )
    (args.output_dir / "production_threshold.json").write_text(
        json.dumps({
            "selected_threshold": optimal_t,
            "source": "train_v11_may8_blend",
        }, indent=2),
        encoding="utf-8",
    )

    # ── Evaluation ──
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    eval_out: dict = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "bundle": str(args.output_dir),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "may8_split": {
            "train_n": len(may8_train),
            "holdout_n": len(may8_holdout),
            "train_frac": args.may8_frac_train,
            "repeat": args.may8_repeat,
        },
        "training_meta": data_meta,
        "optimal_threshold_train": optimal_t,
        "tests": {},
    }

    tests = [
        ("may8_holdout_20pct", may8_holdout),
        ("may8_full_test_parquet", pd.read_parquet(MAY8_PATH)),
    ]
    for tname, tpath in [
        ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet"),
        ("public_test", tpm.TEST_DIR / "public_test_features.parquet"),
        ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
    ]:
        if tpath.is_file():
            tests.append((tname, pd.read_parquet(tpath)))

    for name, df in tests:
        block = eval_labeled(name, df, rf, feature_cols, transform_meta)
        eval_out["tests"][name] = block
        if "bot_recall_pct" in block:
            print(f"  {name:24s} bot_recall={block['bot_recall_pct']}%  "
                  f"human_fpr={block.get('human_fpr_pct')}%  mean={block['mean_score']}")
        elif "human_fpr_pct" in block:
            print(f"  {name:24s} human_fpr={block['human_fpr_pct']}%")

    rd = eval_real_distribution(rf, feature_cols, transform_meta, args.real_dist_dir)
    eval_out["tests"]["real_distribution"] = rd
    print(f"\n  real_distribution: n={rd.get('n')}  mean={rd.get('mean_score')}  "
          f"in[0.5,1]={rd.get('pct_scores_0.5_1.0')}%  pred@0.5={rd.get('pct_pred_bot_0.5')}%")
    if rd.get("corr_vs_logged_risk") is not None:
        print(f"    logged_risk mean={rd.get('logged_risk_mean')}  "
              f"logged@0.5={rd.get('logged_pct_ge_0.5')}%  "
              f"corr={rd.get('corr_vs_logged_risk')}")

    summary = {
        "trained_at": eval_out["trained_at"],
        "n_features": len(feature_cols),
        "n_training_samples": data_meta["total"],
        "data_sources": data_meta.get("sources", {}),
        "may8_blend": {
            "train_base": data_meta.get("may8_train_base"),
            "train_effective": data_meta.get("may8_train_effective"),
            "repeat": args.may8_repeat,
            "holdout_n": len(may8_holdout),
        },
        "selected_threshold": optimal_t,
        "test_results": eval_out["tests"],
    }
    (args.output_dir / "retrain_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    args.eval_json.parent.mkdir(parents=True, exist_ok=True)
    args.eval_json.write_text(json.dumps(eval_out, indent=2), encoding="utf-8")

    print(f"\n[done] bundle → {args.output_dir}")
    print(f"[done] eval   → {args.eval_json}")
    print(f"\nDeploy: POKER44_MINER_MODEL_PATH={args.output_dir / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
