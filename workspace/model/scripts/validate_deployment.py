#!/usr/bin/env python3
"""Smoke-test v11 deployment: .env paths, bundle artifacts, miner-equivalent load + score."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))

BUNDLE = REPO / "workspace/model/artifacts/model_bundle_v11_prod"
ENV_FILE = REPO / ".env"
HOLDOUT = REPO / "workspace/hybrid/dataset/test/may8_holdout_20pct.parquet"


def load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    msg = f"  [{mark}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=" * 70)
    print("DEPLOYMENT VALIDATION — v11 prod")
    print("=" * 70)
    all_ok = True

    env = load_dotenv(ENV_FILE)
    keys = [
        "POKER44_MINER_MODEL_BUNDLE_DIR",
        "POKER44_MINER_MODEL_PATH",
        "POKER44_MINER_TRANSFORM_META_PATH",
        "POKER44_PRODUCTION_THRESHOLD_JSON",
        "POKER44_DYNAMIC_THRESHOLD",
        "POKER44_MINER_REQUIRE_MODEL",
        "POKER44_MINER_OTHER_ONLY",
    ]
    print("\n1) .env configuration")
    for k in keys:
        v = env.get(k, os.environ.get(k, ""))
        all_ok &= check(k, bool(v), v or "(missing)")
    dyn = env.get("POKER44_DYNAMIC_THRESHOLD", "0")
    all_ok &= check("POKER44_DYNAMIC_THRESHOLD=0 (static)", dyn == "0", f"got {dyn!r}")
    other = env.get("POKER44_MINER_OTHER_ONLY", "1")
    all_ok &= check("POKER44_MINER_OTHER_ONLY=0 (model mode)", other == "0", f"got {other!r}")

    print("\n2) Artifact paths exist")
    paths = {
        "bundle_dir": Path(env.get("POKER44_MINER_MODEL_BUNDLE_DIR", str(BUNDLE))),
        "model": Path(env.get("POKER44_MINER_MODEL_PATH", str(BUNDLE / "model.joblib"))),
        "transform": Path(env.get("POKER44_MINER_TRANSFORM_META_PATH", str(BUNDLE / "transform_meta.json"))),
        "threshold": Path(env.get("POKER44_PRODUCTION_THRESHOLD_JSON", str(BUNDLE / "production_threshold.json"))),
    }
    for label, p in paths.items():
        all_ok &= check(label, p.is_file() or (label == "bundle_dir" and p.is_dir()), str(p))

    bundle = paths["bundle_dir"].resolve()
    for fname in ("model.joblib", "feature_cols.json", "transform_meta.json", "production_threshold.json", "retrain_summary.json"):
        all_ok &= check(f"bundle/{fname}", (bundle / fname).is_file())

    print("\n3) Threshold (miner static path)")
    thr_payload = json.loads(paths["threshold"].read_text(encoding="utf-8"))
    thr = float(thr_payload.get("selected_threshold", 0.5))
    all_ok &= check("selected_threshold", abs(thr - 0.55) < 1e-6, f"{thr}")

    print("\n4) Model + features load")
    import joblib
    import numpy as np
    import pandas as pd

    import train_production_model as tpm  # noqa: E402

    model = joblib.load(paths["model"])
    feat_cols = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))["feature_cols"]
    all_ok &= check("feature count", len(feat_cols) == 11, str(len(feat_cols)))
    transform_meta = json.loads(paths["transform"].read_text(encoding="utf-8"))
    clip_keys = set(transform_meta.get("clip_bounds", {}).keys())
    missing = [c for c in feat_cols if c not in clip_keys]
    all_ok &= check("transform_meta clip_bounds", len(missing) == 0, str(missing) or "ok")

    print("\n5) Holdout inference (may8 20% feature parquet)")
    if HOLDOUT.is_file():
        df = pd.read_parquet(HOLDOUT)
        miss = [c for c in feat_cols if c not in df.columns]
        all_ok &= check("holdout has feature cols", len(miss) == 0, str(miss) or "ok")
        X_raw = df[feat_cols].astype(float).values
        X_t = tpm.apply_transform(X_raw, feat_cols, transform_meta)
        probs = model.predict_proba(X_t)[:, 1]
        y = df["label"].values if "label" in df.columns else None
        preds = (probs >= thr).astype(int)
        all_ok &= check("scored rows", len(probs) == len(df), f"n={len(probs)} mean={probs.mean():.3f}")
        all_ok &= check("predictions in [0,1]", bool((probs >= 0).all() and (probs <= 1).all()))
        if y is not None:
            bot = y == 1
            human = y == 0
            recall = float((preds[bot] == 1).mean()) if bot.any() else 0.0
            fpr = float((preds[human] == 1).mean()) if human.any() else 0.0
            all_ok &= check("holdout bot recall @0.55", recall >= 0.85, f"{recall:.1%}")
            all_ok &= check("holdout human FPR @0.55", fpr <= 0.05, f"{fpr:.1%}")
    else:
        print("  [SKIP] holdout parquet missing")

    print("\n6) run_miner.sh preflight")
    run_sh = REPO / "scripts/miner/run/run_miner.sh"
    all_ok &= check("run_miner.sh exists", run_sh.is_file())
    py = os.environ.get("MINER_PYTHON") or env.get("SHARED_MINER_VENV", "")
    if py and not py.endswith("python"):
        py = str(Path(py) / "bin/python")
    if not py or not Path(py).is_file():
        py = sys.executable
    import subprocess

    r = subprocess.run(["bash", "-n", str(run_sh)], capture_output=True, text=True)
    all_ok &= check("bash -n run_miner.sh", r.returncode == 0, r.stderr.strip() or "ok")
    r2 = subprocess.run([py, "-c", "import joblib, sklearn; print('ok')"], capture_output=True, text=True)
    all_ok &= check(f"{py} joblib+sklearn", r2.returncode == 0, r2.stderr.strip() or "ok")

    print("\n7) Miner init sequence (mirrors neurons/miner.py, no bittensor)")
    inference_thr = 0.5
    bundle_dir = paths["bundle_dir"].resolve()
    if paths["threshold"].is_file():
        inference_thr = float(
            json.loads(paths["threshold"].read_text(encoding="utf-8")).get("selected_threshold", 0.5)
        )
    prod_env = env.get("POKER44_PRODUCTION_THRESHOLD_JSON", "")
    if prod_env:
        p = Path(prod_env)
        if p.is_file():
            inference_thr = float(
                json.loads(p.read_text(encoding="utf-8")).get("selected_threshold", inference_thr)
            )
    all_ok &= check("miner _inference_threshold", abs(inference_thr - 0.55) < 1e-6, f"{inference_thr}")
    fc_path = bundle_dir / "feature_cols.json"
    if fc_path.is_file():
        miner_feats = json.loads(fc_path.read_text(encoding="utf-8")).get("feature_cols", [])
        all_ok &= check("miner feature_cols.json", len(miner_feats) == 11)
    tm_path = Path(env.get("POKER44_MINER_TRANSFORM_META_PATH", str(bundle_dir / "transform_meta.json")))
    all_ok &= check("miner transform_meta resolves", tm_path.is_file(), str(tm_path))
    remap = env.get("POKER44_SCORE_REMAP_PATH", "")
    all_ok &= check("score remap disabled", not remap.strip(), remap or "empty")
    salt = env.get("POKER44_FEATURE_SALT", "")
    all_ok &= check("feature salt disabled", not salt.strip(), salt or "empty")

    print("\n8) deploy_v11_prod.sh (validate-only, no retrain)")
    deploy = REPO / "workspace/model/deploy_v11_prod.sh"
    all_ok &= check("deploy script executable", os.access(deploy, os.X_OK))
    for f in ("model.joblib", "feature_cols.json", "transform_meta.json", "production_threshold.json", "retrain_summary.json"):
        all_ok &= check(f"deploy would pass: {f}", (bundle / f).is_file())

    print("\n" + "=" * 70)
    if all_ok:
        print("OVERALL: PASS — deployment ready")
        return 0
    print("OVERALL: FAIL — fix items above before restart")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
