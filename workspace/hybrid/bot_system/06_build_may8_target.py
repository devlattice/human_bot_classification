"""Build a target fingerprint JSON from May-8 gold bots (optionally blended
with the live unlabeled bot fingerprint).

With ``--hard-only``, keeps only May-8 bots that a hold-out RF scores below
``--threshold`` (the 140 "hard" passive bots we fail to catch at 0.5).

Output schema is intentionally identical to ``02_discover_profile.py`` so
``03_match_profiles.py --fp <out>`` can consume it directly.

Writes:
    workspace/hybrid/bot_system/data/may8_target_fingerprint.json
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

import train_production_model as tpm  # noqa: E402

DEFAULT_GOLD = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
DEFAULT_BUNDLE = REPO_ROOT / "workspace" / "hybrid" / "model_bundle"
DEFAULT_LIVE_FP = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "live_bot_fingerprint.json"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "may8_target_fingerprint.json"
DEFAULT_RANKING = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "may8_feature_ranking.json"

META_COLS = {"label", "date", "chunk_idx", "source"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--live-fp", type=Path, default=DEFAULT_LIVE_FP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--date", type=str, default="2026-05-08",
                   help="Gold date(s) to treat as the target. Substring match.")
    p.add_argument("--blend", type=float, default=0.7,
                   help="Weight for May-8 mean (rest goes to live fingerprint, "
                        "if available). 1.0 = pure May-8.")
    p.add_argument(
        "--hard-only",
        action="store_true",
        help="Fingerprint only May-8 bots with hold-out score < --threshold.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Optional bundle for hard-bot split. Default: hold-out RF on gold \\ May-8 "
        "(honest 140-hard split). Production bundle is in-sample and under-counts hard bots.",
    )
    p.add_argument(
        "--use-production-bundle",
        action="store_true",
        help="Score hard bots with --bundle instead of hold-out RF (not recommended).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ranking",
        type=Path,
        default=DEFAULT_RANKING,
        help="may8_feature_ranking.json — sets match_feature_cols + feature_weights for 03.",
    )
    return p.parse_args()


def _match_cols_and_weights(
    target: pd.DataFrame,
    ranking_path: Path,
    *,
    min_auc_hard: float = 0.55,
    top_extra: int = 35,
) -> tuple[list[str], dict[str, float]]:
    """Columns + weights that best separate hard May-8 bots from humans."""
    cols: list[str] = []
    weights: dict[str, float] = {}
    if ranking_path.is_file():
        data = json.loads(ranking_path.read_text(encoding="utf-8"))
        for c in data.get("recommended_focus") or []:
            if c in target.columns:
                cols.append(c)
        feat_by_name = {f["feature"]: f for f in data.get("features") or []}
        for f in sorted(data.get("features") or [], key=lambda x: -x.get("auc_hard_vs_may8_human", 0)):
            name = f.get("feature")
            if not name or name in cols:
                continue
            if float(f.get("auc_hard_vs_may8_human", 0)) < min_auc_hard:
                continue
            if name in target.columns:
                cols.append(name)
            if len(cols) >= top_extra + len(data.get("recommended_focus") or []):
                break
        for c in cols:
            auc = float(feat_by_name.get(c, {}).get("auc_hard_vs_may8_human", min_auc_hard))
            weights[c] = max(auc, min_auc_hard)
        print(f"[ranking] match cols={len(cols)} from {ranking_path.name}")
    if not cols:
        cols = [c for c in tpm.ROBUST_FEATURES if c in target.columns]
        weights = {c: 1.0 for c in cols}
        print("[ranking] fallback: ROBUST_FEATURES for match cols")
    return cols, weights


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _hard_may8_bots(
    gold: pd.DataFrame,
    date: str,
    threshold: float,
    bundle: Path | None,
    seed: int,
    *,
    use_production_bundle: bool,
) -> pd.DataFrame:
    """Return May-8 bot rows with hold-out proba < threshold."""
    may8 = gold[gold["date"].astype(str).str.contains(date) & (gold["label"] == 1)].copy()
    feature_cols = [c for c in tpm.ROBUST_FEATURES if c in may8.columns]

    model_path = (bundle / "model.joblib") if bundle is not None else None
    if use_production_bundle and model_path is not None and model_path.is_file():
        import joblib

        rf = joblib.load(model_path)
        transform_meta = json.loads((bundle / "transform_meta.json").read_text(encoding="utf-8"))
        print(f"[hard-only] scoring with bundle {bundle}")
    else:
        train_gold = gold[~gold["date"].astype(str).str.contains(date)]
        parts_h = [train_gold[train_gold["label"] == 0][feature_cols]]
        parts_b = [train_gold[train_gold["label"] == 1][feature_cols]]
        zen_path = tpm.ZENODO_PATH
        if zen_path.is_file():
            zen = pd.read_parquet(zen_path)
            n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
            parts_h.append(zen.sample(n=n, random_state=seed)[feature_cols])
        Xh = pd.concat(parts_h, ignore_index=True)
        Xb = pd.concat(parts_b, ignore_index=True)
        X_raw = pd.concat([Xh, Xb], ignore_index=True).values
        y = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xb))])
        X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, feature_cols)
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(X_t, y)
        print("[hard-only] scoring with inline hold-out RF (no bundle)")

    Xt = tpm.apply_transform(may8[feature_cols].values, feature_cols, transform_meta)
    may8["proba"] = rf.predict_proba(Xt)[:, 1]
    hard = may8[may8["proba"] < threshold].copy()
    print(
        f"[hard-only] bots={len(may8)} hard={len(hard)} easy={(may8['proba'] >= threshold).sum()} "
        f"threshold={threshold}"
    )
    return hard


def main() -> int:
    args = parse_args()
    gold = pd.read_parquet(args.gold)
    target = gold[gold["date"].str.contains(args.date) & (gold["label"] == 1)]
    if args.hard_only:
        bundle = args.bundle if args.bundle is not None else DEFAULT_BUNDLE
        target = _hard_may8_bots(
            gold,
            args.date,
            args.threshold,
            bundle,
            args.seed,
            use_production_bundle=bool(args.use_production_bundle),
        )
    if target.empty:
        print(f"[error] no rows match date~{args.date} label=1 in {args.gold}")
        return 1
    cols = numeric_cols(target)
    means = target[cols].mean()
    stds = target[cols].std().replace(0, 1e-6)
    print(f"[may8] n={len(target)} dim={len(cols)}")

    blend = float(np.clip(args.blend, 0.0, 1.0))
    if blend < 1.0 and args.live_fp.is_file():
        live = json.loads(args.live_fp.read_text())
        live_means = live.get("feature_means", {})
        live_stds = live.get("feature_stds", {})
        n_overlap = 0
        for c in cols:
            if c in live_means and c in live_stds:
                means[c] = blend * means[c] + (1 - blend) * float(live_means[c])
                stds[c] = blend * stds[c] + (1 - blend) * float(live_stds[c])
                n_overlap += 1
        print(f"[blend] live_fp={args.live_fp.name}  blend={blend:.2f} "
              f"may8/live; overlap_cols={n_overlap}")
    else:
        print(f"[blend] blend={blend:.2f} (no blending applied)")

    match_cols, feature_weights = _match_cols_and_weights(target, args.ranking)

    payload = {
        "source": "may8_hard_gold_bot" if args.hard_only else "may8_gold_bot",
        "date_match": args.date,
        "hard_only": bool(args.hard_only),
        "hard_threshold": float(args.threshold) if args.hard_only else None,
        "n_bot": int(len(target)),
        "blend_may8": blend,
        "feature_cols": cols,
        "feature_means": {c: float(means[c]) for c in cols},
        "feature_stds": {c: float(stds[c]) for c in cols},
        "match_feature_cols": match_cols,
        "feature_weights": {c: float(feature_weights[c]) for c in match_cols},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[done] {args.out}  cols={len(cols)}  match_cols={len(match_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
