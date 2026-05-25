"""Hold-out A/B: ROBUST_FEATURES (54) vs May-8 recommended_focus (~25).

Trains RandomForest with May-8 gold excluded (same recipe as 09_train_and_check),
then evaluates on May-8, May-7, and held-out test parquets.

Output:
  workspace/hybrid/bot_system/data/may8_feature_ab_results.json

Usage:
  python workspace/hybrid/scripts/eval_may8_feature_ab.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))

import train_production_model as tpm  # noqa: E402

DEFAULT_RANKING = (
    REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_feature_ranking.json"
)
DEFAULT_OUT = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_feature_ab_results.json"
THRESH = 0.5


def load_or_none(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_file() else None


def intersect_features(cols: list[str], dfs: list[pd.DataFrame | None]) -> list[str]:
    avail = set(cols)
    for df in dfs:
        if df is not None and len(df):
            avail &= set(df.columns)
    return [c for c in cols if c in avail]


def evaluate_block(
    name: str,
    df: pd.DataFrame | None,
    feature_cols: list[str],
    rf: RandomForestClassifier,
    transform_meta: dict,
    *,
    label: int,
) -> dict | None:
    if df is None or df.empty:
        return None
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        return {"name": name, "error": f"missing {len(miss)} features", "n": int(len(df))}
    Xt = tpm.apply_transform(df[feature_cols].values, feature_cols, transform_meta)
    p = rf.predict_proba(Xt)[:, 1]
    out: dict = {
        "name": name,
        "n": int(len(df)),
        "mean_score": round(float(p.mean()), 4),
        "median_score": round(float(np.median(p)), 4),
        "score_p90": round(float(np.percentile(p, 90)), 4),
    }
    if label == 0:
        out["fpr_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
        out["n_flagged"] = int((p >= THRESH).sum())
    else:
        out["recall_pct"] = round(float((p >= THRESH).mean()) * 100, 3)
        out["n_detected"] = int((p >= THRESH).sum())
        for t in (0.15, 0.25, 0.35):
            out[f"recall_at_{t:.2f}_pct"] = round(float((p >= t).mean()) * 100, 2)
    return out


def train_holdout(
    feature_cols: list[str],
    *,
    seed: int,
) -> tuple[RandomForestClassifier, dict, dict]:
    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        raise FileNotFoundError(tpm.GOLD_PATH)

    gold_train = gold_full[~gold_full["date"].astype(str).str.contains("2026-05-08")]
    zen = load_or_none(tpm.ZENODO_PATH)
    pub = load_or_none(tpm.PUBLIC_PATH)
    fs = load_or_none(tpm.FULL_SPECTRUM_PATH)
    ac = load_or_none(tpm.ACPC_BOT_PATH)

    dfs = [gold_train, zen, pub, fs, ac]
    feature_cols = intersect_features(feature_cols, dfs)

    rng = np.random.RandomState(seed)
    parts_h, parts_b = [], []
    src: dict[str, int] = {}

    parts_h.append(gold_train[gold_train["label"] == 0][feature_cols])
    parts_b.append(gold_train[gold_train["label"] == 1][feature_cols])
    src["gold_human"] = int((gold_train["label"] == 0).sum())
    src["gold_bot"] = int((gold_train["label"] == 1).sum())

    if zen is not None:
        n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(zen.sample(n=n, random_state=rng)[feature_cols])
        src["zenodo_human"] = n
    if pub is not None:
        pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
        n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
        parts_h.append(pub_up.sample(n=n, random_state=rng)[feature_cols])
        src["public_human"] = n
    if fs is not None:
        n = min(len(fs), tpm.BOT_SAMPLE_CAP)
        parts_b.append(fs.sample(n=n, random_state=rng)[feature_cols])
        src["full_spectrum_bot"] = n
    if ac is not None:
        n = min(len(ac), tpm.BOT_SAMPLE_CAP)
        parts_b.append(ac.sample(n=n, random_state=rng)[feature_cols])
        src["acpc_bot"] = n

    Xh = pd.concat(parts_h, ignore_index=True)
    Xb = pd.concat(parts_b, ignore_index=True)
    X_raw = pd.concat([Xh, Xb], ignore_index=True).values
    y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])

    X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_t, y)
    train_p = rf.predict_proba(X_t)[:, 1]
    train_meta = {
        "n_features": len(feature_cols),
        "train_n": int(len(y)),
        "train_human": int(len(Xh)),
        "train_bot": int(len(Xb)),
        "train_auc": round(float(roc_auc_score(y, train_p)), 4),
        "train_acc": round(float(accuracy_score(y, (train_p >= THRESH).astype(int))), 4),
        "sources": src,
    }
    train_meta["feature_cols"] = feature_cols
    return rf, transform_meta, train_meta


def eval_all_blocks(
    rf: RandomForestClassifier,
    transform_meta: dict,
    feature_cols: list[str],
    gold_full: pd.DataFrame,
    *,
    may8_hard_bots: pd.DataFrame | None,
    may8_easy_bots: pd.DataFrame | None,
) -> dict[str, dict | None]:
    may8 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-08")]
    may7 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-07")]
    hard = may8_hard_bots if may8_hard_bots is not None else may8.iloc[0:0]
    easy = may8_easy_bots if may8_easy_bots is not None else may8.iloc[0:0]

    blocks: dict[str, dict | None] = {
        "may8_human": evaluate_block(
            "may8_human", may8[may8["label"] == 0], feature_cols, rf, transform_meta, label=0
        ),
        "may8_bot": evaluate_block(
            "may8_bot", may8[may8["label"] == 1], feature_cols, rf, transform_meta, label=1
        ),
        "may8_hard_bot": evaluate_block(
            "may8_hard_bot", hard, feature_cols, rf, transform_meta, label=1
        ),
        "may8_easy_bot": evaluate_block(
            "may8_easy_bot", easy, feature_cols, rf, transform_meta, label=1
        ),
        "may7_human": evaluate_block(
            "may7_human", may7[may7["label"] == 0], feature_cols, rf, transform_meta, label=0
        ),
        "may7_bot": evaluate_block(
            "may7_bot", may7[may7["label"] == 1], feature_cols, rf, transform_meta, label=1
        ),
        "zenodo_test": evaluate_block(
            "zenodo_test",
            load_or_none(tpm.TEST_DIR / "zenodo_test_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            label=0,
        ),
        "public_test": evaluate_block(
            "public_test",
            load_or_none(tpm.TEST_DIR / "public_test_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            label=0,
        ),
        "acpc_bot_test": evaluate_block(
            "acpc_bot_test",
            load_or_none(tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            label=1,
        ),
        "wsop_stress": evaluate_block(
            "wsop_stress",
            load_or_none(tpm.TEST_DIR / "wsop_stress_features.parquet"),
            feature_cols,
            rf,
            transform_meta,
            label=0,
        ),
    }
    if blocks["may8_bot"] and blocks["may8_hard_bot"]:
        blocks["may8_bot"]["n_hard_bots"] = int(blocks["may8_hard_bot"].get("n", 0))
        blocks["may8_bot"]["n_easy_bots"] = int(blocks["may8_easy_bot"].get("n", 0) if blocks["may8_easy_bot"] else 0)
    return blocks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_table(variants: list[dict]) -> None:
    keys = [
        ("may8_bot", "recall_pct", "May-8 bot recall"),
        ("may8_hard_bot", "recall_pct", "May-8 hard recall"),
        ("may8_human", "fpr_pct", "May-8 human FPR"),
        ("zenodo_test", "fpr_pct", "Zenodo test FPR"),
        ("public_test", "fpr_pct", "Public test FPR"),
        ("acpc_bot_test", "recall_pct", "ACPC bot recall"),
    ]
    print("\n" + "=" * 78)
    print("HOLD-OUT A/B @ threshold 0.5")
    print("=" * 78)
    hdr = f"{'metric':<22}" + "".join(f"{v['name']:>18}" for v in variants)
    print(hdr)
    for block, field, label in keys:
        row = f"{label:<22}"
        for v in variants:
            b = (v.get("evaluations") or {}).get(block) or {}
            val = b.get(field)
            row += f"{val if val is not None else 'n/a':>18}"
        print(row)
    print("=" * 78)


def main() -> int:
    args = parse_args()
    if not args.ranking.is_file():
        print(f"[error] missing ranking: {args.ranking}")
        return 1

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    focus_cols = list(ranking.get("recommended_focus") or [])
    if not focus_cols:
        print("[error] recommended_focus empty in ranking JSON")
        return 1

    gold_full = load_or_none(tpm.GOLD_PATH)
    if gold_full is None:
        print(f"[error] missing {tpm.GOLD_PATH}")
        return 1

    variants_spec = [
        ("robust_54", list(tpm.ROBUST_FEATURES)),
        ("may8_focus_25", focus_cols),
    ]

    # Fixed hard/easy split from reference hold-out RF (robust 54, all numeric train)
    ref_rf, ref_tm, _ = train_holdout(list(tpm.ROBUST_FEATURES), seed=args.seed)
    ref_cols = intersect_features(list(tpm.ROBUST_FEATURES), [gold_full])
    may8 = gold_full[gold_full["date"].astype(str).str.contains("2026-05-08")]
    may8_b = may8[may8["label"] == 1].copy()
    Xt_ref = tpm.apply_transform(may8_b[ref_cols].values, ref_cols, ref_tm)
    ref_scores = ref_rf.predict_proba(Xt_ref)[:, 1]
    may8_hard = may8_b.loc[ref_scores < THRESH].copy()
    may8_easy = may8_b.loc[ref_scores >= THRESH].copy()
    print(
        f"\n[ref split] robust hold-out: hard={len(may8_hard)} easy={len(may8_easy)} "
        f"(same as rank_may8_features)"
    )

    variants: list[dict] = []
    t0 = time.time()
    for name, cols in variants_spec:
        print(f"\n[train] variant={name}  requested_features={len(cols)}")
        rf, tm, train_meta = train_holdout(cols, seed=args.seed)
        feat = list(train_meta["feature_cols"])
        print(
            f"  features={len(feat)}  train_n={train_meta['train_n']}  "
            f"AUC={train_meta['train_auc']}"
        )
        evals = eval_all_blocks(
            rf, tm, feat, gold_full,
            may8_hard_bots=may8_hard,
            may8_easy_bots=may8_easy,
        )
        variants.append({
            "name": name,
            "feature_cols": feat,
            "train": train_meta,
            "evaluations": evals,
        })

    # Pick winner on May-8 bot recall with zenodo FPR <= 2%
    def score_variant(v: dict) -> tuple[float, float]:
        e = v["evaluations"]
        recall = float((e.get("may8_bot") or {}).get("recall_pct") or 0.0)
        zen_fpr = float((e.get("zenodo_test") or {}).get("fpr_pct") or 99.0)
        penalty = 50.0 if zen_fpr > 2.0 else 0.0
        return (recall - penalty, -zen_fpr)

    ranked = sorted(variants, key=score_variant, reverse=True)
    winner = ranked[0]["name"]
    robust = next(v for v in variants if v["name"] == "robust_54")
    focus = next(v for v in variants if v["name"] == "may8_focus_25")
    delta_recall = (
        float((focus["evaluations"].get("may8_bot") or {}).get("recall_pct") or 0)
        - float((robust["evaluations"].get("may8_bot") or {}).get("recall_pct") or 0)
    )

    payload = {
        "version": 1,
        "ts": datetime.utcnow().isoformat() + "Z",
        "threshold": THRESH,
        "feature_pipeline": ranking.get("feature_pipeline"),
        "ranking_source": str(args.ranking),
        "elapsed_sec": round(time.time() - t0, 1),
        "variants": variants,
        "may8_hard_easy_split": {
            "reference_model": "robust_54_holdout",
            "threshold": THRESH,
            "n_hard_bots": int(len(may8_hard)),
            "n_easy_bots": int(len(may8_easy)),
        },
        "comparison": {
            "winner_by_may8_recall_zen_fpr_cap_2pct": winner,
            "may8_recall_delta_focus_minus_robust_pct": round(delta_recall, 3),
        },
        "gates": {
            "may8_recall_target_pct": 80.0,
            "zenodo_fpr_cap_pct": 2.0,
            "public_fpr_cap_pct": 3.0,
            "acpc_recall_target_pct": 90.0,
        },
    }
    for v in variants:
        e = v["evaluations"]
        m8 = (e.get("may8_bot") or {}).get("recall_pct")
        zf = (e.get("zenodo_test") or {}).get("fpr_pct")
        v["pass_may8_80"] = m8 is not None and m8 >= 80.0
        v["pass_zen_fpr_2"] = zf is None or zf <= 2.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_table(variants)
    print(f"\n[done] {args.out}")
    print(
        f"  winner={winner}  may8 recall delta (focus-robust)={delta_recall:+.2f} pp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
