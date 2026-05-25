"""Shared May-8 profile validation (KS generated vs hard gold)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "workspace" / "hybrid" / "scripts"))

import train_production_model as tpm  # type: ignore


def load_hard_may8_bots(gold: pd.DataFrame, date: str = "2026-05-08") -> pd.DataFrame:
    return gold[gold["date"].astype(str).str.contains(date) & (gold["label"] == 1)].copy()


def validate_vs_may8_hard(
    generated: pd.DataFrame,
    gold: pd.DataFrame,
    fingerprint: dict[str, Any],
    *,
    date: str = "2026-05-08",
) -> dict[str, Any]:
    """Compare generated chunk features to all May-8 gold bots (proxy for profile fit)."""
    may8_bot = load_hard_may8_bots(gold, date)
    cols = fingerprint.get("match_feature_cols") or [
        c for c in tpm.ROBUST_FEATURES if c in generated.columns and c in may8_bot.columns
    ]
    weights = fingerprint.get("feature_weights") or {}

    rows: list[dict] = []
    weighted_ks: list[float] = []
    for c in cols:
        if c not in generated.columns or c not in may8_bot.columns:
            continue
        a = generated[c].fillna(0).astype(float).values
        b = may8_bot[c].fillna(0).astype(float).values
        if len(a) < 3 or len(b) < 3:
            continue
        ks = float(ks_2samp(a, b).statistic)
        w = float(weights.get(c, 1.0))
        rows.append({
            "feature": c,
            "ks_gen_vs_hard": round(ks, 4),
            "gen_mean": round(float(np.mean(a)), 6),
            "hard_mean": round(float(np.mean(b)), 6),
            "weight": round(w, 4),
        })
        weighted_ks.extend([ks] * max(1, int(w * 10)))

    rows.sort(key=lambda r: r["ks_gen_vs_hard"])
    median_ks = float(np.median([r["ks_gen_vs_hard"] for r in rows])) if rows else 1.0
    mean_ks = float(np.mean([r["ks_gen_vs_hard"] for r in rows])) if rows else 1.0
    wmedian = float(np.median(weighted_ks)) if weighted_ks else median_ks

    return {
        "n_generated": int(len(generated)),
        "n_may8_bot_gold": int(len(may8_bot)),
        "n_features": len(rows),
        "median_ks": round(median_ks, 4),
        "mean_ks": round(mean_ks, 4),
        "weighted_median_ks": round(wmedian, 4),
        "worst_5": rows[-5:] if len(rows) >= 5 else rows,
        "best_5": rows[:5],
        "features": rows,
    }


def profile_reached(summary: dict[str, Any], *, max_median_ks: float, max_weighted_median_ks: float) -> bool:
    med = summary.get("median_ks")
    wmed = summary.get("weighted_median_ks")
    if med is None:
        return False
    if float(med) > max_median_ks:
        return False
    if wmed is not None and float(wmed) > max_weighted_median_ks:
        return False
    return True
