"""Shared helpers for feature_selection_v3 (LOOCV scoring, ranking, pair rescue)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_production_model as tpm  # noqa: E402
from shell_progress import ShellProgress, iter_progress  # noqa: E402

HYBRID_DIR = REPO_ROOT / "workspace" / "hybrid"
TRAIN_DIR = HYBRID_DIR / "dataset" / "train"
TEST_DIR = HYBRID_DIR / "dataset" / "test"

BANNED_PREFIXES = ("other_ratio", "n_actions")
CORR_THRESHOLD = 0.95
META_COLS = frozenset({"label", "source", "date", "chunk_idx", "chunk_hash"})
THRESHOLD = 0.5

FPR_ZENODO_CAP = 0.01
FPR_PUBLIC_CAP = 0.01
FPR_GOLD_CAP = 0.02


def default_rf_kwargs(seed: int = 42, *, fast: bool = False) -> dict:
    if fast:
        return {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "min_samples_split": 2,
            "random_state": int(seed),
            "n_jobs": -1,
            "class_weight": "balanced",
        }
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 15,
        "min_samples_split": 2,
        "random_state": int(seed),
        "n_jobs": -1,
        "class_weight": "balanced",
    }


def load_gold_train() -> pd.DataFrame:
    path = TRAIN_DIR / "gold_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_train_tables(feature_cols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load train parquets (selection fit only — never test/)."""
    if feature_cols:
        return tpm.load_datasets(feature_cols)
    # probe columns from gold
    gold = load_gold_train()
    cols = prefilter_candidates(gold)
    return tpm.load_datasets(cols)


def intersect_train_columns(
    candidates: list[str],
    datasets: dict[str, pd.DataFrame],
    *,
    required: tuple[str, ...] = ("gold", "zenodo", "acpc_bot"),
) -> list[str]:
    """Keep features present in required train tables (gold + zenodo + acpc)."""
    avail = set(candidates)
    for name in required:
        if name not in datasets:
            continue
        avail &= set(datasets[name].columns)
    return [c for c in candidates if c in avail]


def prefilter_candidates(gold: pd.DataFrame) -> list[str]:
    """Phase 0: banned, constant, correlated dedup."""
    feats = [
        c
        for c in gold.columns
        if c not in META_COLS
        and not any(str(c).startswith(p) for p in BANNED_PREFIXES)
        and pd.api.types.is_numeric_dtype(gold[c])
    ]
    stds = gold[feats].std()
    feats = [f for f in feats if float(stds[f]) >= 1e-8]

    corr = gold[feats].corr().abs()
    drop: set[str] = set()
    for i, fi in enumerate(feats):
        if fi in drop:
            continue
        for j in range(i + 1, len(feats)):
            fj = feats[j]
            if fj in drop:
                continue
            if float(corr.iloc[i, j]) > CORR_THRESHOLD:
                drop.add(fj)

    return [f for f in feats if f not in drop]


def build_diverse_ranked(
    ranked_scores: list[tuple[str, float]],
    gold: pd.DataFrame,
) -> list[str]:
    """Greedy order: high score first, skip |r|>0.95 vs already picked."""
    ordered = [f for f, _ in sorted(ranked_scores, key=lambda x: -x[1])]
    picked: list[str] = []
    for f in ordered:
        if not picked:
            picked.append(f)
            continue
        ok = True
        for g in picked:
            r = gold[[f, g]].corr().iloc[0, 1]
            if np.isfinite(r) and abs(float(r)) > CORR_THRESHOLD:
                ok = False
                break
        if ok:
            picked.append(f)
    return picked


def _bot_recall_fpr(y_true: np.ndarray, proba: np.ndarray, thresh: float = THRESHOLD) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    bots = y == 1
    humans = y == 0
    recall = float((p[bots] >= thresh).mean()) if bots.any() else 0.0
    fpr = float((p[humans] >= thresh).mean()) if humans.any() else 0.0
    return recall, fpr


class SelectionContext:
    """Cached train data slices and external subsamples for repeated LOOCV."""

    def __init__(
        self,
        gold_train: pd.DataFrame,
        candidates: list[str],
        seed: int = 42,
    ) -> None:
        self.gold = gold_train
        self.candidates = list(candidates)
        self.seed = int(seed)
        self.dates = sorted(gold_train["date"].astype(str).unique())
        self.rng = np.random.RandomState(seed)

        self.datasets = tpm.load_datasets(self.candidates)
        self.zen_sub_full = self._subsample_human("zenodo", 1500)
        self.pub_sub_full = self._subsample_human("public", 800)

    def external_for(self, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return self._build_external_arrays(features)

    def zen_public_for(self, features: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
        zen = pub = None
        if self.zen_sub_full is not None:
            df = pd.DataFrame(self.zen_sub_full, columns=self.candidates)
            zen = df[features].values
        if self.pub_sub_full is not None:
            df = pd.DataFrame(self.pub_sub_full, columns=self.candidates)
            pub = df[features].values
        return zen, pub

    def _build_external_arrays(self, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        parts_h: list[np.ndarray] = []
        parts_b: list[np.ndarray] = []

        if "zenodo" in self.datasets:
            zen = self.datasets["zenodo"]
            n = min(len(zen), tpm.HUMAN_SAMPLE_CAP)
            parts_h.append(zen.sample(n=n, random_state=self.rng)[cols].values)

        if "public" in self.datasets:
            pub = self.datasets["public"]
            pub_up = pd.concat([pub] * tpm.PUBLIC_OVERSAMPLE, ignore_index=True)
            n = min(len(pub_up), tpm.HUMAN_SAMPLE_CAP)
            parts_h.append(pub_up.sample(n=n, random_state=self.rng)[cols].values)

        if "full_spectrum" in self.datasets:
            fs = self.datasets["full_spectrum"]
            n = min(len(fs), tpm.BOT_SAMPLE_CAP)
            parts_b.append(fs.sample(n=n, random_state=self.rng)[cols].values)

        if "acpc_bot" in self.datasets:
            ac = self.datasets["acpc_bot"]
            n = min(len(ac), tpm.BOT_SAMPLE_CAP)
            parts_b.append(ac.sample(n=n, random_state=self.rng)[cols].values)

        ext_h = np.vstack(parts_h) if parts_h else np.empty((0, len(cols)))
        ext_b = np.vstack(parts_b) if parts_b else np.empty((0, len(cols)))
        return ext_h, ext_b

    def _subsample_human(self, name: str, n: int) -> np.ndarray | None:
        if name not in self.datasets:
            return None
        df = self.datasets[name]
        nn = min(len(df), n)
        return df.sample(n=nn, random_state=self.rng)[self.candidates].values


def evaluate_feature_subset(
    ctx: SelectionContext,
    features: list[str],
    rf_kwargs: dict,
    *,
    thresh: float = THRESHOLD,
) -> dict[str, Any]:
    """LOOCV on gold + FPR on zenodo/public subsamples; production-style transforms."""
    if not features:
        return {"valid": False, "reason": "empty_features"}

    gold = ctx.gold
    dates = ctx.dates
    ext_h, ext_b = ctx.external_for(features)
    zen_sub, pub_sub = ctx.zen_public_for(features)

    day_bot_recalls: list[float] = []
    day_human_fprs: list[float] = []

    for test_date in dates:
        test = gold[gold["date"].astype(str) == test_date]
        train_gold = gold[gold["date"].astype(str) != test_date]

        train_X = np.vstack([
            train_gold[features].values,
            ext_h,
            ext_b,
        ])
        train_y = np.concatenate([
            train_gold["label"].values.astype(int),
            np.zeros(len(ext_h)),
            np.ones(len(ext_b)),
        ])

        X_t, transform_meta = tpm.fit_transform_pipeline(train_X, features)
        test_X_t = tpm.apply_transform(test[features].values, features, transform_meta)

        rf = RandomForestClassifier(**rf_kwargs)
        rf.fit(X_t, train_y)
        proba = rf.predict_proba(test_X_t)[:, 1]
        rec, fpr = _bot_recall_fpr(test["label"].values, proba, thresh)
        day_bot_recalls.append(rec)
        day_human_fprs.append(fpr)

    min_recall = float(min(day_bot_recalls)) if day_bot_recalls else 0.0
    mean_recall = float(np.mean(day_bot_recalls)) if day_bot_recalls else 0.0
    max_fpr_gold = float(max(day_human_fprs)) if day_human_fprs else 0.0

    # Aux FPR: fit on all gold train + external
    train_X_full = np.vstack([gold[features].values, ext_h, ext_b])
    train_y_full = np.concatenate([
        gold["label"].values.astype(int),
        np.zeros(len(ext_h)),
        np.ones(len(ext_b)),
    ])
    X_full, tm = tpm.fit_transform_pipeline(train_X_full, features)
    rf_full = RandomForestClassifier(**rf_kwargs)
    rf_full.fit(X_full, train_y_full)

    fpr_zen = 0.0
    if zen_sub is not None and len(zen_sub):
        pz = rf_full.predict_proba(tpm.apply_transform(zen_sub, features, tm))[:, 1]
        fpr_zen = float((pz >= thresh).mean())

    fpr_pub = 0.0
    if pub_sub is not None and len(pub_sub):
        pp = rf_full.predict_proba(tpm.apply_transform(pub_sub, features, tm))[:, 1]
        fpr_pub = float((pp >= thresh).mean())

    acpc_recall = 0.0
    if "acpc_bot" in ctx.datasets:
        ac = ctx.datasets["acpc_bot"]
        n = min(len(ac), 2000)
        sub = ac.sample(n=n, random_state=ctx.rng)[features]
        pa = rf_full.predict_proba(tpm.apply_transform(sub.values, features, tm))[:, 1]
        if (ac["label"].values == 1).any():
            acpc_recall = float((pa >= thresh).mean())

    max_fpr = max(fpr_zen, fpr_pub, max_fpr_gold)
    score = (
        0.50 * min_recall
        + 0.30 * mean_recall
        + 0.15 * acpc_recall
        - 0.20 * max_fpr
    )

    valid = (
        fpr_zen <= FPR_ZENODO_CAP
        and fpr_pub <= FPR_PUBLIC_CAP
        and max_fpr_gold <= FPR_GOLD_CAP
    )

    return {
        "valid": valid,
        "score": float(score),
        "min_recall": min_recall,
        "mean_recall": mean_recall,
        "acpc_recall": acpc_recall,
        "max_fpr_gold": max_fpr_gold,
        "fpr_zenodo": fpr_zen,
        "fpr_public": fpr_pub,
        "n_features": len(features),
        "per_day_recall": dict(zip(dates, [round(x, 4) for x in day_bot_recalls])),
    }


def rank_univariate(
    ctx: SelectionContext,
    candidates: list[str],
    *,
    fast: bool = True,
    quiet: bool = False,
) -> list[tuple[str, float]]:
    """Rank each feature: gold LOOCV + acpc recall − FPR (full train mix FIT)."""
    kw = default_rf_kwargs(ctx.seed, fast=fast)
    scores: list[tuple[str, float]] = []
    for f in iter_progress(candidates, desc="[rank] LOOCV/univariate", disable=quiet):
        m = evaluate_feature_subset(ctx, [f], kw, thresh=THRESHOLD)
        s = float(m["score"]) if m.get("valid") else float(m["min_recall"]) - 0.1
        scores.append((f, s))
    return scores


def run_pair_rescue(
    ctx: SelectionContext,
    ranked: list[str],
    *,
    x_base: int = 30,
    borderline: int = 25,
    top_a: int = 10,
    tau_pair: float = 0.02,
    tau_weak_pair: float = 0.03,
    max_promotions: int = 10,
    fast: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Phase 0b: promote borderline features via pairwise LOOCV."""
    kw = default_rf_kwargs(ctx.seed, fast=fast)
    x_base = min(x_base, len(ranked))
    core = ranked[:x_base]

    # Baseline singles in borderline
    border_feats = ranked[x_base : x_base + borderline]
    promotions: list[dict[str, Any]] = []
    promoted_set: set[str] = set()

    def try_promote(feat: str, reason: str, score: float, partner: str | None = None) -> None:
        if feat in promoted_set or len(promoted_set) >= max_promotions:
            return
        if feat in core:
            return
        promoted_set.add(feat)
        promotions.append({
            "feature": feat,
            "reason": reason,
            "score": round(score, 4),
            "partner": partner,
        })

    # Strong + weak
    top_a_feats = ranked[: min(top_a, len(ranked))]
    sw_border = border_feats[:15]
    for c in iter_progress(sw_border, desc="[pair_rescue] strong+weak", disable=quiet):
        solo_c = evaluate_feature_subset(ctx, [c], kw)["score"]
        best_gain = 0.0
        best_a: str | None = None
        for a in top_a_feats:
            if a == c:
                continue
            m = evaluate_feature_subset(ctx, [a, c], kw)
            gain = m["score"] - solo_c
            if gain > best_gain:
                best_gain = gain
                best_a = a
        if best_gain >= tau_pair:
            try_promote(c, "strong_weak_pair", solo_c + best_gain, best_a)

    # Weak + weak
    best_border_single = 0.0
    if border_feats:
        for f in iter_progress(
            border_feats[:10], desc="[pair_rescue] border solo", disable=quiet
        ):
            best_border_single = max(
                best_border_single, evaluate_feature_subset(ctx, [f], kw)["score"]
            )

    ww_pairs = [
        (border_feats[i], border_feats[j])
        for i in range(len(border_feats))
        for j in range(i + 1, len(border_feats))
    ]
    for c, d in iter_progress(ww_pairs, desc="[pair_rescue] weak+weak", disable=quiet):
        m = evaluate_feature_subset(ctx, [c, d], kw)
        if m["score"] >= best_border_single + tau_weak_pair:
            try_promote(c, "weak_weak_pair", m["score"], d)
            try_promote(d, "weak_weak_pair", m["score"], c)

    pool = core + [f for f in ranked if f in promoted_set and f not in core]
    # preserve ranked order for rest
    seen = set(pool)
    for f in ranked:
        if f not in seen:
            pool.append(f)
            seen.add(f)

    return {
        "x_base": x_base,
        "borderline": borderline,
        "core": core,
        "promotions": promotions,
        "promoted": sorted(promoted_set),
        "pool": pool,
    }


def composite_score(metrics: dict[str, Any], *, pool_penalty_x: int | None = None) -> float:
    s = float(metrics.get("score", -1e6))
    if pool_penalty_x is not None:
        s -= 0.002 * pool_penalty_x
    return s


def eval_test_parquet(
    path: Path,
    features: list[str],
    rf: RandomForestClassifier,
    transform_meta: dict,
    *,
    label_col: str = "label",
    thresh: float = THRESHOLD,
) -> dict[str, Any]:
    df = pd.read_parquet(path)
    miss = [c for c in features if c not in df.columns]
    if miss:
        return {"error": f"missing {len(miss)} cols", "n": len(df)}
    Xt = tpm.apply_transform(df[features].values, features, transform_meta)
    proba = rf.predict_proba(Xt)[:, 1]
    y = df[label_col].values.astype(int)
    out: dict[str, Any] = {"n": int(len(df)), "mean_score": round(float(proba.mean()), 4)}
    if (y == 0).any():
        out["human_fpr_pct"] = round(float((proba[y == 0] >= thresh).mean()) * 100, 3)
    if (y == 1).any():
        out["bot_recall_pct"] = round(float((proba[y == 1] >= thresh).mean()) * 100, 2)
    return out


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
