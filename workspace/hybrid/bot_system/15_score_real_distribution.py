"""Score unlabeled validator chunks from ``workspace/dataset/real_distribution``.

Loads a model bundle (RandomForest + feature_cols + transform_meta), runs the
validator-delivered chunks: ``aggregate_chunk_from_miner_payload`` (same as
``neurons/miner.py``) → ``train_production_model.apply_transform`` →
``predict_proba``.

Writes:
    workspace/hybrid/bot_system/data/real_distribution_model_scores.csv
    workspace/hybrid/bot_system/data/real_distribution_model_scores_summary.txt

Compares each line's logged ``risk_score`` (previous deployment) to the new
``model_score`` (probability of bot class).
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
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid" / "scripts"))

import joblib  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid"))
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402

import train_production_model as tpm  # type: ignore  # noqa: E402

DEFAULT_BUNDLE = REPO_ROOT / "workspace" / "hybrid" / "model_bundle_v6_passive_robust_micro"
DEFAULT_IN = REPO_ROOT / "workspace" / "dataset" / "real_distribution"
DEFAULT_CSV = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "real_distribution_model_scores.csv"
DEFAULT_SUMMARY = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "real_distribution_model_scores_summary.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_IN)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rf = joblib.load(args.bundle / "lgbm_student.joblib")
    meta = json.loads((args.bundle / "feature_cols.json").read_text())
    feature_cols: list[str] = meta["feature_cols"]
    transform_meta = json.loads((args.bundle / "transform_meta.json").read_text())

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        print(f"[error] no jsonl in {args.input_dir}")
        return 1

    rows_out: list[dict] = []
    bad = 0
    for fp in files:
        src = fp.name
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
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
                if not raw:
                    bad += 1
                    continue
                miss = [c for c in feature_cols if c not in raw]
                if miss:
                    bad += 1
                    continue
                Xv = np.asarray(
                    tpm.apply_transform(
                        np.asarray([raw[c] for c in feature_cols], dtype=np.float64)[None, :],
                        feature_cols,
                        transform_meta,
                    ),
                    dtype=np.float64,
                )
                score = float(rf.predict_proba(Xv)[0, 1])
                rs = obj.get("risk_score")
                rs_f = float(rs) if isinstance(rs, (int, float)) else None
                pred = score >= 0.5
                rows_out.append({
                    "source_file": src,
                    "line": line_no,
                    "chunk_hash": obj.get("chunk_hash", ""),
                    "risk_score_logged": rs_f,
                    "model_score": score,
                    "pred_bot_ge_0.5": pred,
                })

    if not rows_out:
        print("[error] no scored rows")
        return 2

    df = pd.DataFrame(rows_out)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Per unique chunk_hash: mean logged vs mean model (dedup view)
    g = df.groupby("chunk_hash", dropna=False).agg(
        n_lines=("model_score", "count"),
        risk_score_logged_mean=("risk_score_logged", "mean"),
        risk_score_logged_std=("risk_score_logged", "std"),
        model_score_mean=("model_score", "mean"),
        model_score_std=("model_score", "std"),
        pred_bot_frac=("pred_bot_ge_0.5", "mean"),
    ).reset_index()
    dedup_csv = args.out_csv.with_name(args.out_csv.stem + "_by_chunk_hash.csv")
    g.to_csv(dedup_csv, index=False)

    # Summary text
    lines: list[str] = []
    lines.append(f"bundle: {args.bundle}")
    lines.append(f"input_dir: {args.input_dir}")
    lines.append(f"raw_lines_scored: {len(df)}  parse_fail_or_skip: {bad}")
    lines.append("")

    rs = df["risk_score_logged"].dropna()
    ms = df["model_score"]
    lines.append("=== All lines (every log row) ===")
    lines.append(f"  risk_score_logged: mean={rs.mean():.4f}  median={rs.median():.4f}  "
                 f"p90={rs.quantile(0.9):.4f}  >=0.5: {(rs >= 0.5).mean()*100:.1f}%  "
                 f">=0.35: {(rs >= 0.35).mean()*100:.1f}%  >=0.20: {(rs >= 0.20).mean()*100:.1f}%")
    lines.append(f"  model_score (v6):  mean={ms.mean():.4f}  median={ms.median():.4f}  "
                 f"p90={ms.quantile(0.9):.4f}  >=0.5: {(ms >= 0.5).mean()*100:.1f}%  "
                 f">=0.35: {(ms >= 0.35).mean()*100:.1f}%  >=0.20: {(ms >= 0.20).mean()*100:.1f}%")
    if len(rs) >= 10:
        sub = df.dropna(subset=["risk_score_logged"])
        corr = float(np.corrcoef(sub["risk_score_logged"].values, sub["model_score"].values)[0, 1])
        lines.append(f"  Pearson corr(logged, model): {corr:.4f}")
    lines.append(f"  mean(model - logged) per line: {(df['model_score'] - df['risk_score_logged']).mean():+.4f}")
    lines.append("")

    lines.append("=== Dedup by chunk_hash (mean per hash) ===")
    lines.append(f"  unique chunk_hash: {len(g)}")
    lines.append(f"  risk_score_logged_mean: mean={g['risk_score_logged_mean'].mean():.4f}  "
                 f"median={g['risk_score_logged_mean'].median():.4f}  "
                 f">=0.5: {(g['risk_score_logged_mean'] >= 0.5).mean()*100:.1f}%")
    lines.append(f"  model_score_mean:        mean={g['model_score_mean'].mean():.4f}  "
                 f"median={g['model_score_mean'].median():.4f}  "
                 f">=0.5: {(g['model_score_mean'] >= 0.5).mean()*100:.1f}%")
    lines.append(f"  pred_bot_frac (mean of line-level bot preds per hash): "
                 f"mean={g['pred_bot_frac'].mean()*100:.1f}%")

    lines.append("")
    lines.append("Sample rows (highest model_score):")
    top = df.nlargest(8, "model_score")
    for _, r in top.iterrows():
        lines.append(
            f"  model={r['model_score']:.4f}  logged={r['risk_score_logged']}  "
            f"pred_bot={r['pred_bot_ge_0.5']}  hash={str(r['chunk_hash'])[:16]}…  {r['source_file']}"
        )
    lines.append("")
    lines.append("Sample rows (lowest model_score — most 'human'):")
    bot = df.nsmallest(8, "model_score")
    for _, r in bot.iterrows():
        lines.append(
            f"  model={r['model_score']:.4f}  logged={r['risk_score_logged']}  "
            f"pred_bot={r['pred_bot_ge_0.5']}  hash={str(r['chunk_hash'])[:16]}…  {r['source_file']}"
        )

    txt = "\n".join(lines)
    args.out_summary.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[csv] {args.out_csv}")
    print(f"[csv] {dedup_csv}")
    print(f"[summary] {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
