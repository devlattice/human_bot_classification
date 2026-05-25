"""Cluster unlabeled validator chunks and fingerprint the bot cluster.

- Extracts features for logged validator→miner chunks (already miner-visible;
  uses ``aggregate_chunk_from_miner_payload``, same as live miner scoring).
- Standard-scales features, runs KMeans(k=2). Picks the cluster with the
  HIGHER risk_score_mean as the "bot" cluster (uses logged scores only as a
  tiebreaker — not as a label).
- Writes:
    workspace/hybrid/bot_system/data/unlabeled_features.parquet
    workspace/hybrid/bot_system/data/live_bot_fingerprint.json
        { feature_means, feature_stds, n, n_bot, n_human, picked_cluster_id }
- Also reports KS distance vs May-8 gold bot to give a sanity check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid"))

from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

DEFAULT_IN = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "unlabeled_unique.jsonl"
DEFAULT_GOLD = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
DEFAULT_FEAT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "unlabeled_features.parquet"
DEFAULT_FP_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "live_bot_fingerprint.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--feat-out", type=Path, default=DEFAULT_FEAT_OUT)
    p.add_argument("--fp-out", type=Path, default=DEFAULT_FP_OUT)
    p.add_argument("--k", type=int, default=2)
    return p.parse_args()


def extract_features(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("chunk")
            if not isinstance(chunk, list):
                continue
            try:
                row = aggregate_chunk_from_miner_payload(chunk)
            except Exception:
                continue
            if not row:
                continue
            row["chunk_hash"] = obj.get("chunk_hash", "")
            row["risk_score_mean"] = obj.get("risk_score_mean")
            row["n_observed"] = obj.get("n_observed", 0)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = extract_features(args.input)
    print(f"[features] extracted {len(df)} rows × {df.shape[1]} cols")
    args.feat_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.feat_out, index=False)
    print(f"[out] {args.feat_out}")

    meta_cols = {"chunk_hash", "risk_score_mean", "n_observed"}
    numeric_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].fillna(0.0).astype(np.float64).values
    Xs = RobustScaler(quantile_range=(5, 95)).fit_transform(X)

    km = KMeans(n_clusters=args.k, random_state=0, n_init=20).fit(Xs)
    labels = km.labels_
    df["cluster"] = labels

    # Pick bot cluster: the one with the higher mean risk_score (tiebreaker).
    score_by_cluster = (
        df.groupby("cluster")["risk_score_mean"].mean().to_dict()
    )
    bot_cluster = int(max(score_by_cluster, key=score_by_cluster.get))
    print(f"[cluster] sizes={dict(pd.Series(labels).value_counts())}  "
          f"score_means={ {k: round(v, 4) for k, v in score_by_cluster.items()} }  "
          f"picked bot_cluster={bot_cluster}")

    bot_mask = labels == bot_cluster
    bot_df = df[bot_mask][numeric_cols]
    means = bot_df.mean()
    stds = bot_df.std().replace(0, 1e-6)

    payload = {
        "n_unique_chunks": int(len(df)),
        "n_bot": int(bot_mask.sum()),
        "n_human": int((~bot_mask).sum()),
        "picked_cluster": bot_cluster,
        "cluster_risk_score_means": {str(k): float(v) for k, v in score_by_cluster.items()},
        "feature_means": {c: float(means[c]) for c in numeric_cols},
        "feature_stds": {c: float(stds[c]) for c in numeric_cols},
        "feature_cols": numeric_cols,
    }
    args.fp_out.parent.mkdir(parents=True, exist_ok=True)
    args.fp_out.write_text(json.dumps(payload, indent=2))
    print(f"[fingerprint] {args.fp_out}  n_bot={payload['n_bot']} n_human={payload['n_human']}")

    # Sanity: KS vs May-8 gold bot
    gold = pd.read_parquet(args.gold)
    may8_bot = gold[(gold["date"] == "2026-05-08") & (gold["label"] == 1)]
    overlap = [c for c in numeric_cols if c in may8_bot.columns]
    ks_vals = []
    for c in overlap:
        a = may8_bot[c].dropna().values
        b = bot_df[c].dropna().values
        if len(a) >= 10 and len(b) >= 10:
            try:
                ks_vals.append(ks_2samp(a, b).statistic)
            except Exception:
                continue
    ks_arr = np.asarray(ks_vals)
    print(f"\n[sanity] KS(live_bot_cluster, may8_gold_bot)  "
          f"mean={ks_arr.mean():.4f}  median={np.median(ks_arr):.4f}  n_feat={len(ks_arr)}")
    print("[hint] lower = more similar. Target after generation: ks_mean < 0.3")

    # Show top behavioral knobs the fingerprint is anchored on
    print("\n[top fingerprint anchors] highest |z| vs May-8 gold bot")
    z = []
    for c in overlap:
        if c in payload["feature_means"]:
            a_mu = float(may8_bot[c].mean())
            a_sd = float(may8_bot[c].std() or 1e-6)
            z.append((c, abs((payload["feature_means"][c] - a_mu) / a_sd)))
    z.sort(key=lambda r: -r[1])
    for c, val in z[:15]:
        print(f"  {c:<40} |z|={val:.2f}  live={payload['feature_means'][c]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
