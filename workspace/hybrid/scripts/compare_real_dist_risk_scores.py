"""Compare real_distribution logged risk_score (54-feat prod) vs new model scores.

Usage:
  python workspace/hybrid/scripts/compare_real_dist_risk_scores.py \\
    --bundle-v11 workspace/hybrid/model_bundle_v3_11 \\
    --bundle-v54 workspace/hybrid/model_bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

import train_production_model as tpm  # noqa: E402
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402

REAL_DIST = REPO / "workspace" / "dataset" / "real_distribution"
DEFAULT_OUT = REPO / "workspace" / "hybrid/bot_system/data/real_dist_risk_vs_v11.json"
THRESH = 0.5


def load_bundle(bundle: Path) -> tuple:
    for name in ("model.joblib", "lgbm_student.joblib"):
        mp = bundle / name
        if mp.is_file():
            rf = joblib.load(mp)
            break
    else:
        raise FileNotFoundError(bundle)
    cols = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))["feature_cols"]
    tm = json.loads((bundle / "transform_meta.json").read_text(encoding="utf-8"))
    static_t = None
    for nm in ("production_threshold.json", "retrain_summary.json"):
        p = bundle / nm
        if p.is_file():
            try:
                static_t = float(json.loads(p.read_text(encoding="utf-8")).get("selected_threshold"))
            except Exception:
                pass
            if static_t is not None:
                break
    return rf, cols, tm, static_t


def score_chunks(
    real_dir: Path,
    rf,
    cols: list[str],
    tm: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    logged: list[float] = []
    model: list[float] = []
    meta = {"parse_fail": 0, "missing_feat": 0, "n_lines": 0, "files": {}}

    for fp in sorted(real_dir.glob("*.jsonl")):
        n_ok = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta["n_lines"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    meta["parse_fail"] += 1
                    continue
                chunk = obj.get("chunk")
                rs = obj.get("risk_score")
                if not isinstance(chunk, list) or not isinstance(rs, (int, float)):
                    meta["parse_fail"] += 1
                    continue
                try:
                    raw = aggregate_chunk_from_miner_payload(chunk)
                except Exception:
                    meta["parse_fail"] += 1
                    continue
                if not raw or any(c not in raw for c in cols):
                    meta["missing_feat"] += 1
                    continue
                Xv = tpm.apply_transform(
                    np.asarray([raw[c] for c in cols], dtype=np.float64)[None, :],
                    cols,
                    tm,
                )
                p = float(rf.predict_proba(Xv)[0, 1])
                logged.append(float(rs))
                model.append(p)
                n_ok += 1
        meta["files"][fp.name] = n_ok

    return np.asarray(logged), np.asarray(model), meta


def band_stats(scores: np.ndarray, label: str) -> dict:
    s = scores[np.isfinite(scores)]
    if len(s) == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(np.median(s)), 4),
        "p10": round(float(np.percentile(s, 10)), 4),
        "p90": round(float(np.percentile(s, 90)), 4),
        "pct_in_0.5_1.0": round(float(((s >= 0.5) & (s <= 1.0)).mean()) * 100, 2),
        "pct_ge_0.5": round(float((s >= 0.5).mean()) * 100, 2),
        "pct_lt_0.1": round(float((s < 0.1).mean()) * 100, 2),
    }


def compare_pair(logged: np.ndarray, model: np.ndarray, name: str) -> dict:
    n = min(len(logged), len(model))
    lg, md = logged[:n], model[:n]
    corr = float(np.corrcoef(lg, md)[0, 1]) if n > 2 else None
    diff = md - lg
    return {
        "name": name,
        "n": n,
        "corr": round(corr, 4) if corr is not None else None,
        "mean_abs_diff": round(float(np.abs(diff).mean()), 4),
        "mean_diff_model_minus_logged": round(float(diff.mean()), 4),
        "pct_model_higher": round(float((md > lg).mean()) * 100, 2),
        "logged": band_stats(lg, "logged_risk_54"),
        "model": band_stats(md, f"rescored_{name}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    ap.add_argument("--bundle-v11", type=Path, default=REPO / "workspace/hybrid/model_bundle_v3_11")
    ap.add_argument("--bundle-v54", type=Path, default=REPO / "workspace/hybrid/model_bundle")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("Loading bundles...")
    rf11, cols11, tm11, st11 = load_bundle(args.bundle_v11)
    rf54, cols54, tm54, st54 = load_bundle(args.bundle_v54)

    print("Scoring real_distribution with 11-feat bundle...")
    logged, sc11, meta11 = score_chunks(args.real_dist_dir, rf11, cols11, tm11)
    print(f"  scored n={len(sc11)}  parse_fail={meta11['parse_fail']}")

    print("Scoring same chunks with 54-feat bundle (fresh rescore)...")
    _, sc54, meta54 = score_chunks(args.real_dist_dir, rf54, cols54, tm54)
    print(f"  scored n={len(sc54)}")

    # logged risk_score is from 54-feat miner at collection time
    cmp_log_vs_11 = compare_pair(logged, sc11, "v11")
    cmp_log_vs_54 = compare_pair(logged, sc54, "v54_rescore")
    cmp_11_vs_54 = compare_pair(sc11, sc54, "v11_vs_v54_rescore")

    out = {
        "real_dist_dir": str(args.real_dist_dir),
        "n_aligned": int(min(len(logged), len(sc11), len(sc54))),
        "note": (
            "risk_score in JSONL = scores from previous 54-feature miner at log time. "
            "v54_rescore = same 54-feat model re-applied to chunks today (sanity). "
            "v11 = new 11-feature model_bundle_v3_11."
        ),
        "logged_risk_score_only": band_stats(logged, "jsonl_risk_score"),
        "compare_logged_vs_v11": cmp_log_vs_11,
        "compare_logged_vs_v54_rescore": cmp_log_vs_54,
        "compare_v11_vs_v54_rescore": cmp_11_vs_54,
        "threshold_0.5": {
            "logged_risk_pct_bot": cmp_log_vs_11["logged"]["pct_ge_0.5"],
            "v11_pct_bot": cmp_log_vs_11["model"]["pct_ge_0.5"],
            "v54_rescore_pct_bot": cmp_log_vs_54["model"]["pct_ge_0.5"],
        },
        "files": meta11["files"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("REAL DISTRIBUTION: logged risk_score (54) vs models")
    print("=" * 72)
    for key in ("logged_risk_score_only",):
        b = out[key]
        print(f"\nJSONL risk_score (54-feat at log time):")
        print(f"  mean={b['mean']}  median={b['median']}  in[0.5,1]={b['pct_in_0.5_1.0']}%  >=0.5={b['pct_ge_0.5']}%  <0.1={b['pct_lt_0.1']}%")

    for block in (cmp_log_vs_11, cmp_log_vs_54, cmp_11_vs_54):
        print(f"\n{block['name']}: corr={block['corr']}  mean_diff={block['mean_diff_model_minus_logged']}  |diff|={block['mean_abs_diff']}")
        print(f"  logged/left:  mean={block['logged']['mean']}  >=0.5={block['logged']['pct_ge_0.5']}%")
        print(f"  model/right:  mean={block['model']['mean']}  >=0.5={block['model']['pct_ge_0.5']}%")

    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
