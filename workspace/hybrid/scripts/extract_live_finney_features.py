"""Extract chunk features from unlabeled real_distribution JSONL (no labels).

Splits rows into:
  train/live_finney_unlabeled.parquet  — used in FS Phase 0 (variance / column coverage)
  test/live_finney_monitor.parquet     — held-out live slice (monitor only, never in Optuna score)

Usage:
  python workspace/hybrid/scripts/extract_live_finney_features.py \\
    --jsonl workspace/dataset/real_distribution/1_0.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402

OUT_TRAIN = REPO / "workspace/hybrid/dataset/train/live_finney_unlabeled.parquet"
OUT_MONITOR = REPO / "workspace/hybrid/dataset/test/live_finney_monitor.parquet"
PROFILE_OUT = REPO / "workspace/hybrid/bot_system/data/live_profile.json"


def iter_jsonl_chunks(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "chunk" not in obj:
            continue
        rows.append(obj)
    return rows


def extract_rows(objs: list[dict], *, source: str) -> pd.DataFrame:
    records: list[dict] = []
    for i, obj in enumerate(objs):
        ch = obj["chunk"]
        if isinstance(ch, str):
            ch = json.loads(ch)
        agg = aggregate_chunk_from_miner_payload(ch)
        agg["chunk_hash"] = obj.get("chunk_hash") or f"row_{i}"
        agg["risk_score_logged"] = float(obj["risk_score"]) if "risk_score" in obj else np.nan
        agg["source"] = source
        records.append(agg)
    return pd.DataFrame(records)


def profile_frame(df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    meta = [c for c in ("chunk_hash", "risk_score_logged", "source") if c in df.columns]
    feats = feature_cols or [c for c in df.columns if c not in meta]
    stats: dict = {
        "n_chunks": int(len(df)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": {},
    }
    for c in feats:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].astype(float)
        stats["features"][c] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "p10": round(float(s.quantile(0.10)), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p90": round(float(s.quantile(0.90)), 6),
        }
    if "risk_score_logged" in df.columns:
        rs = df["risk_score_logged"].dropna().astype(float)
        if len(rs):
            stats["risk_score_logged"] = {
                "mean": round(float(rs.mean()), 4),
                "pct_ge_0.5": round(100 * float((rs >= 0.5).mean()), 2),
                "pct_ge_0.55": round(100 * float((rs >= 0.55).mean()), 2),
            }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True, help="real_distribution NDJSON log")
    ap.add_argument("--train-out", type=Path, default=OUT_TRAIN)
    ap.add_argument("--monitor-out", type=Path, default=OUT_MONITOR)
    ap.add_argument("--profile-out", type=Path, default=PROFILE_OUT)
    ap.add_argument("--train-frac", type=float, default=0.5, help="Fraction for FS coverage pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-name", type=str, default="live_finney")
    args = ap.parse_args()

    if not args.jsonl.is_file():
        print(f"[error] missing {args.jsonl}")
        return 1

    objs = iter_jsonl_chunks(args.jsonl)
    if not objs:
        print("[error] no chunk rows in jsonl")
        return 1

    df = extract_rows(objs, source=args.source_name)
    print(f"Extracted {len(df)} chunks, {len(df.columns)} columns from {args.jsonl}")

    # Unlabeled: no stratify; random split for coverage vs monitor
    if len(df) < 2:
        train_df, mon_df = df, df.iloc[0:0]
    else:
        train_df, mon_df = train_test_split(
            df, train_size=args.train_frac, random_state=args.seed
        )

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.monitor_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.train_out, index=False)
    mon_df.to_parquet(args.monitor_out, index=False)

    prof = {
        "source_jsonl": str(args.jsonl.resolve()),
        "train_parquet": str(args.train_out.resolve()),
        "monitor_parquet": str(args.monitor_out.resolve()),
        "train_n": int(len(train_df)),
        "monitor_n": int(len(mon_df)),
        "train_profile": profile_frame(train_df),
        "monitor_profile": profile_frame(mon_df),
    }
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(json.dumps(prof, indent=2), encoding="utf-8")

    print(f"Saved train (unlabeled): {args.train_out}  n={len(train_df)}")
    print(f"Saved monitor:         {args.monitor_out}  n={len(mon_df)}")
    print(f"Saved profile:         {args.profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
