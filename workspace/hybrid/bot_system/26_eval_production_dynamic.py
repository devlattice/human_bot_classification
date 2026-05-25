"""Simulate production miner: model scores + fixed 0.5 vs dynamic threshold per request.

Evaluates as if validators send batched chunk requests (test parquets, May-8 hold-out,
unlabeled real_distribution JSONL).

Usage:
  python workspace/hybrid/bot_system/26_eval_production_dynamic.py \\
    --bundle workspace/hybrid/model_bundle_may8_reflect
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
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

import train_production_model as tpm  # noqa: E402
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402
from poker44.miner.dynamic_threshold import DynamicThresholdPolicy, ThresholdConfig  # noqa: E402

REAL_DIST = REPO / "workspace" / "dataset" / "real_distribution"
DEFAULT_BUNDLE = REPO / "workspace" / "hybrid" / "model_bundle_may8_reflect"
DEFAULT_OUT = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "production_dynamic_eval.json"
MAY8_DATE = "2026-05-08"
FIXED_T = 0.5


def load_bundle(bundle: Path) -> tuple[RandomForestClassifier, list[str], dict, float | None]:
    model_path = None
    for name in ("model.joblib", "lgbm_student.joblib"):
        p = bundle / name
        if p.is_file():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"no model.joblib in {bundle}")
    rf = joblib.load(model_path)
    cols = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))["feature_cols"]
    tm = json.loads((bundle / "transform_meta.json").read_text(encoding="utf-8"))
    static_t: float | None = None
    for name in ("production_threshold.json", "retrain_summary.json"):
        p = bundle / name
        if p.is_file():
            try:
                static_t = float(json.loads(p.read_text(encoding="utf-8")).get("selected_threshold"))
                break
            except Exception:
                pass
    return rf, cols, tm, static_t


def score_df(
    df: pd.DataFrame, cols: list[str], rf: RandomForestClassifier, tm: dict,
) -> np.ndarray:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing {len(miss)} feature cols")
    Xt = tpm.apply_transform(df[cols].values, cols, tm)
    return rf.predict_proba(Xt)[:, 1].astype(np.float64)


def simulate_requests(
    scores: np.ndarray,
    labels: np.ndarray | None,
    *,
    batch_size: int,
    policy: DynamicThresholdPolicy,
    fixed_t: float = FIXED_T,
) -> dict:
    """One validator request = up to batch_size chunks; dynamic threshold per request."""
    n = len(scores)
    policy = DynamicThresholdPolicy(policy.config)  # fresh stream per dataset

    pred_fixed = np.zeros(n, dtype=bool)
    pred_dynamic = np.zeros(n, dtype=bool)
    thresholds: list[float] = []
    modes: list[str] = []

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = scores[start:end].tolist()
        dec = policy.decide_and_observe(batch)
        t = dec.threshold
        thresholds.extend([t] * (end - start))
        modes.extend([dec.mode] * (end - start))
        pred_dynamic[start:end] = scores[start:end] >= t
        pred_fixed[start:end] = scores[start:end] >= fixed_t

    out: dict = {
        "n_chunks": int(n),
        "n_requests": int((n + batch_size - 1) // batch_size),
        "batch_size": batch_size,
        "score_mean": round(float(scores.mean()), 4),
        "score_median": round(float(np.median(scores)), 4),
        "threshold_mean": round(float(np.mean(thresholds)), 4) if thresholds else None,
        "threshold_median": round(float(np.median(thresholds)), 4) if thresholds else None,
        "mode_counts": {str(k): int(v) for k, v in pd.Series(modes).value_counts().items()},
    }

    if labels is not None:
        y = labels.astype(int)
        bot = y == 1
        human = y == 0
        out["fixed_0.5"] = {
            "bot_recall_pct": round(float(pred_fixed[bot].mean()) * 100, 2) if bot.any() else None,
            "human_fpr_pct": round(float(pred_fixed[human].mean()) * 100, 3) if human.any() else None,
        }
        out["dynamic"] = {
            "bot_recall_pct": round(float(pred_dynamic[bot].mean()) * 100, 2) if bot.any() else None,
            "human_fpr_pct": round(float(pred_dynamic[human].mean()) * 100, 3) if human.any() else None,
        }
    else:
        out["fixed_0.5"] = {
            "pred_bot_pct": round(float(pred_fixed.mean()) * 100, 2),
        }
        out["dynamic"] = {
            "pred_bot_pct": round(float(pred_dynamic.mean()) * 100, 2),
        }

    return out


def may8_hard_easy(
    may8_b: pd.DataFrame,
    cols: list[str],
    gold_train: pd.DataFrame,
    zen: pd.DataFrame | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return may8_b.loc[ref_p < FIXED_T], may8_b.loc[ref_p >= FIXED_T]


def eval_real_distribution(
    rf: RandomForestClassifier,
    cols: list[str],
    tm: dict,
    policy_cfg: ThresholdConfig,
    input_dir: Path,
    batch_size: int,
) -> dict:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        return {"error": f"no jsonl in {input_dir}"}

    scores_list: list[float] = []
    logged: list[float] = []
    bad = 0
    file_ids: list[str] = []

    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
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
                if not raw or any(c not in raw for c in cols):
                    bad += 1
                    continue
                Xv = tpm.apply_transform(
                    np.asarray([raw[c] for c in cols], dtype=np.float64)[None, :],
                    cols, tm,
                )
                scores_list.append(float(rf.predict_proba(Xv)[0, 1]))
                rs = obj.get("risk_score")
                if isinstance(rs, (int, float)):
                    logged.append(float(rs))
                file_ids.append(fp.name)

    if not scores_list:
        return {"error": "no scored rows", "parse_fail": bad}

    scores = np.asarray(scores_list, dtype=np.float64)

    overall = simulate_requests(
        scores, None, batch_size=batch_size, policy=DynamicThresholdPolicy(policy_cfg),
    )

    # Per-file requests (validator session = one jsonl file)
    per_file: dict[str, dict] = {}
    for fname in sorted(set(file_ids)):
        idx = [i for i, f in enumerate(file_ids) if f == fname]
        sub_scores = scores[idx]
        pf = simulate_requests(
            sub_scores, None,
            batch_size=batch_size,
            policy=DynamicThresholdPolicy(policy_cfg),
        )
        per_file[fname] = {
            "n": len(idx),
            "fixed_pred_bot_pct": pf["fixed_0.5"].get("pred_bot_pct"),
            "dynamic_pred_bot_pct": pf["dynamic"].get("pred_bot_pct"),
            "threshold_median": pf.get("threshold_median"),
            "mode_counts": pf.get("mode_counts"),
        }

    out = {
        "parse_fail": bad,
        "overall": overall,
        "per_file_sample": dict(list(per_file.items())[:8]),
        "n_files": len(set(file_ids)),
    }
    if len(logged) >= 10:
        lg = np.asarray(logged[: len(scores)])
        out["logged_risk_mean"] = round(float(lg.mean()), 4)
        out["corr_logged_vs_model"] = round(float(np.corrcoef(lg, scores[: len(lg)])[0, 1]), 4)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=20, help="Chunks per simulated validator request.")
    p.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rf, cols, tm, static_t = load_bundle(args.bundle)

    policy_cfg = ThresholdConfig(
        fixed_fallback=FIXED_T,
        static_selected=static_t,
        clamp_min=0.10,
        clamp_max=0.45,
        min_scores=20,
        rolling_window=500,
    )

    gold = pd.read_parquet(tpm.GOLD_PATH)
    gold_train = gold[~gold["date"].astype(str).str.contains(MAY8_DATE)]
    may8 = gold[gold["date"].astype(str).str.contains(MAY8_DATE)]
    may8_b = may8[may8["label"] == 1].copy()
    may8_h = may8[may8["label"] == 0].copy()
    may8_hard, may8_easy = may8_hard_easy(may8_b, cols, gold_train, None, args.seed)

    datasets: list[tuple[str, pd.DataFrame | None, int | None]] = [
        ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet", 0),
        ("public_test", tpm.TEST_DIR / "public_test_features.parquet", 0),
        ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet", 1),
        ("wsop_stress", tpm.TEST_DIR / "wsop_stress_features.parquet", 0),
        ("may8_human", may8_h, 0),
        ("may8_bot_all", may8_b, 1),
        ("may8_hard_bot", may8_hard, 1),
        ("may8_easy_bot", may8_easy, 1),
    ]

    results: dict = {
        "bundle": str(args.bundle),
        "batch_size": args.batch_size,
        "static_selected_fallback": static_t,
        "policy": {
            "clamp": [policy_cfg.clamp_min, policy_cfg.clamp_max],
            "min_scores": policy_cfg.min_scores,
        },
        "datasets": {},
    }

    print("=" * 72)
    print("PRODUCTION SIMULATION (batched validator requests)")
    print(f"  bundle={args.bundle.name}  batch_size={args.batch_size}  static_fallback={static_t}")
    print("=" * 72)

    for name, path_or_df, label in datasets:
        if isinstance(path_or_df, Path):
            if not path_or_df.is_file():
                print(f"\n[{name}] SKIP missing file")
                continue
            df = pd.read_parquet(path_or_df)
        else:
            df = path_or_df
        if df is None or df.empty:
            continue
        scores = score_df(df, cols, rf, tm)
        sim = simulate_requests(
            scores,
            df["label"].values if label is not None else None,
            batch_size=args.batch_size,
            policy=DynamicThresholdPolicy(policy_cfg),
        )
        results["datasets"][name] = sim
        fx = sim["fixed_0.5"]
        dy = sim["dynamic"]
        if label == 1:
            print(
                f"\n[{name}] n={sim['n_chunks']} requests={sim['n_requests']} "
                f"score_med={sim['score_median']}"
            )
            print(
                f"  fixed@0.5   bot_recall={fx.get('bot_recall_pct')}%"
            )
            print(
                f"  dynamic     bot_recall={dy.get('bot_recall_pct')}%  "
                f"thr_med={sim.get('threshold_median')} modes={sim.get('mode_counts')}"
            )
        elif label == 0:
            print(
                f"\n[{name}] n={sim['n_chunks']} "
                f"fixed FPR={fx.get('human_fpr_pct')}%  dynamic FPR={dy.get('human_fpr_pct')}%  "
                f"thr_med={sim.get('threshold_median')}"
            )

    print("\n" + "=" * 72)
    print("REAL DISTRIBUTION (unlabeled, batched like live requests)")
    print("=" * 72)
    rd = eval_real_distribution(rf, cols, tm, policy_cfg, args.real_dist_dir, args.batch_size)
    results["real_distribution"] = rd
    if "error" in rd:
        print(f"  error: {rd['error']}")
    else:
        ov = rd["overall"]
        print(f"  n_scored={ov['n_chunks']} requests={ov['n_requests']} parse_fail={rd.get('parse_fail')}")
        fx = ov.get("fixed_0.5") or {}
        dy = ov.get("dynamic") or {}
        print(
            f"  score mean={ov['score_mean']}  fixed bot%={fx.get('pred_bot_pct')}  "
            f"dynamic bot%={dy.get('pred_bot_pct')}"
        )
        print(f"  dynamic thr_med={ov.get('threshold_median')} modes={ov.get('mode_counts')}")
        if "corr_logged_vs_model" in rd:
            print(f"  corr(logged,model)={rd['corr_logged_vs_model']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
