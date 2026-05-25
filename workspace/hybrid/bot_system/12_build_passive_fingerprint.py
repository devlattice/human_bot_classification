"""Build fingerprint JSON from May-8 gold bots that score < 0.5 (ultra-passive).

Uses the same schema as live_bot_fingerprint.json so 03_match_profiles --fp works.

Writes:
    workspace/hybrid/bot_system/data/passive_hard_fingerprint.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid" / "scripts"))

import joblib  # noqa: E402

import train_production_model as tpm  # type: ignore  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "round_03_bundle"
DEFAULT_GOLD = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "passive_hard_fingerprint.json"

META = {"label", "date", "chunk_idx", "source"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rf = joblib.load(args.bundle / "lgbm_student.joblib")
    feature_cols = json.loads((args.bundle / "feature_cols.json").read_text())["feature_cols"]
    transform_meta = json.loads((args.bundle / "transform_meta.json").read_text())

    gold = pd.read_parquet(args.gold)
    may8 = gold[gold["date"].str.contains("2026-05-08")].copy()
    bots = may8[may8["label"] == 1].copy()
    Xt = tpm.apply_transform(bots[feature_cols].values, feature_cols, transform_meta)
    bots["proba"] = rf.predict_proba(Xt)[:, 1]
    hard = bots[bots["proba"] < args.threshold].copy()
    if hard.empty:
        print("[error] no hard bots; check bundle / gold")
        return 1

    num_cols = [c for c in hard.columns if c not in META and pd.api.types.is_numeric_dtype(hard[c])]
    means = hard[num_cols].mean()
    stds = hard[num_cols].std().replace(0, 1e-6)

    payload = {
        "source": "may8_passive_hard_bots",
        "n_bot": int(len(hard)),
        "n_easy_bots": int((bots["proba"] >= args.threshold).sum()),
        "threshold": args.threshold,
        "bundle": str(args.bundle),
        "feature_cols": num_cols,
        "feature_means": {c: float(means[c]) for c in num_cols},
        "feature_stds": {c: float(stds[c]) for c in num_cols},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[passive-hard] n={len(hard)}  dims={len(num_cols)}  out={args.out}")
    print(f"  proba mean={hard['proba'].mean():.4f}  median={hard['proba'].median():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
