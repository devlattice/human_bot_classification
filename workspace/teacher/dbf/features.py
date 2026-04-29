"""
Distilled Behavior Features (DBF).

Computed from per-hand tabular columns (e.g. feature_45_rb): counts, ratios, street
aggregates, diversity — no raw sequence or identity. Safe for miner-aligned schemas.

All output columns are prefixed with ``dbf_`` so they are easy to spot and to exclude
from legacy feature lists if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-6

# Fixed set appended by add_dbf_columns (stable for manifests / LGBM feature lists).
DBF_COLUMN_NAMES: tuple[str, ...] = (
    "dbf_aggression_intensity",
    "dbf_aggr_minus_fold",
    "dbf_action_diversity_norm",
    "dbf_transition_complexity",
    "dbf_repetition_score",
    "dbf_early_pressure",
    "dbf_late_pressure",
    "dbf_street_pressure_shift",
)


def normalize_dbf_column_subset(dbf_columns: Sequence[str] | None) -> tuple[str, ...]:
    """Return ordered DBF column names to compute; default is all."""
    if dbf_columns is None:
        return DBF_COLUMN_NAMES
    cols = tuple(dbf_columns)
    bad = [c for c in cols if c not in DBF_COLUMN_NAMES]
    if bad:
        raise ValueError(f"dbf_columns contains unknown entries: {bad}")
    if len(set(cols)) != len(cols):
        raise ValueError("dbf_columns must be unique")
    return cols


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _clip_quantile(a: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
    """Winsorize by quantiles computed on the current array (can leak split info)."""
    x = np.asarray(a, dtype=np.float64)
    if x.size == 0:
        return x
    lo = float(np.nanquantile(x, lo_q))
    hi = float(np.nanquantile(x, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)


def _clip_with_bounds(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Winsorize using fixed (train-fitted) bounds."""
    x = np.asarray(a, dtype=np.float64)
    if x.size == 0:
        return x
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)


def _squash_tanh(a: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Bound feature range to [-1, 1] while preserving ordering."""
    x = np.asarray(a, dtype=np.float64)
    s = max(float(scale), EPS)
    return np.tanh(x / s)


def _stabilize_feature(a: np.ndarray, *, clip_bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Robust post-process: winsorize + squash + finite/low-var guard.

    If ``clip_bounds`` is ``(lo, hi)``, clip to those train-fitted bounds.
    If ``clip_bounds`` is None, clip using quantiles of ``a`` only (legacy / single-split use).
    """
    if clip_bounds is None:
        x = _clip_quantile(a)
    else:
        lo, hi = clip_bounds
        x = _clip_with_bounds(a, lo, hi)
    x = _squash_tanh(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    # If a feature degenerates to almost constant, zero it to avoid brittle bias.
    if float(np.nanstd(x)) < 1e-7:
        return np.zeros_like(x, dtype=np.float64)
    return x


def _compute_dbf_raw_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw DBF vectors (pre-winsorize / pre-tanh), row-aligned with ``df``."""
    n = len(df)
    z = np.zeros(n, dtype=np.float64)

    # NOTE:
    # The feature_2 schema does not include raw action-count columns such as
    # count_raise/count_call/actions_on_preflop/etc. The previous DBF formulas
    # silently fell back to zeros for missing inputs, producing all-zero dbf_*.
    # These formulas are intentionally derived only from the keep_features list.
    bet_mean = _num(df["bet_ratio_mean"]) if "bet_ratio_mean" in df.columns else pd.Series(z, index=df.index)
    call_mean = _num(df["call_ratio_mean"]) if "call_ratio_mean" in df.columns else pd.Series(z, index=df.index)
    check_mean = _num(df["check_ratio_mean"]) if "check_ratio_mean" in df.columns else pd.Series(z, index=df.index)
    raise_mean = _num(df["raise_ratio_mean"]) if "raise_ratio_mean" in df.columns else pd.Series(z, index=df.index)
    fold_mean = _num(df["fold_ratio_mean"]) if "fold_ratio_mean" in df.columns else pd.Series(z, index=df.index)
    raise_max = _num(df["raise_ratio_max"]) if "raise_ratio_max" in df.columns else pd.Series(z, index=df.index)

    # Aggression proxy: emphasize raises/bets, penalize call/fold passivity.
    aggression_intensity = (0.8 * raise_mean + 0.6 * bet_mean) - (0.4 * call_mean + 0.8 * fold_mean)
    aggr_minus_fold = raise_max - fold_mean

    # "Diversity": high when action means are close to each other.
    spread = (
        np.abs((bet_mean - call_mean).to_numpy(dtype=np.float64))
        + np.abs((call_mean - check_mean).to_numpy(dtype=np.float64))
        + np.abs((check_mean - raise_mean).to_numpy(dtype=np.float64))
    ) / 3.0
    action_diversity_norm = 1.0 / (1.0 + spread)

    n_streets_mean = _num(df["n_streets_mean"]) if "n_streets_mean" in df.columns else pd.Series(z, index=df.index)
    n_streets_std = _num(df["n_streets_std"]) if "n_streets_std" in df.columns else pd.Series(z, index=df.index)
    transition_complexity = n_streets_mean.to_numpy(dtype=np.float64) + 0.5 * n_streets_std.to_numpy(dtype=np.float64)

    check_std = _num(df["check_ratio_std"]) if "check_ratio_std" in df.columns else pd.Series(z, index=df.index)
    call_std = _num(df["call_ratio_std"]) if "call_ratio_std" in df.columns else pd.Series(z, index=df.index)
    raise_std = _num(df["raise_ratio_std"]) if "raise_ratio_std" in df.columns else pd.Series(z, index=df.index)
    repetition_score = (check_std + call_std - raise_std).to_numpy(dtype=np.float64)

    end_preflop = _num(df["end_preflop_mean"]) if "end_preflop_mean" in df.columns else pd.Series(z, index=df.index)
    end_flop = _num(df["end_flop_mean"]) if "end_flop_mean" in df.columns else pd.Series(z, index=df.index)
    end_turn = _num(df["end_turn_mean"]) if "end_turn_mean" in df.columns else pd.Series(z, index=df.index)
    end_river = _num(df["end_river_mean"]) if "end_river_mean" in df.columns else pd.Series(z, index=df.index)

    # Early vs late pressure from street-level aggregates.
    early_raw = (end_preflop + end_flop).to_numpy(dtype=np.float64)
    late_raw = (end_turn + end_river).to_numpy(dtype=np.float64)
    street_scale = np.abs(early_raw) + np.abs(late_raw) + EPS
    early_pressure = early_raw / street_scale
    late_pressure = late_raw / street_scale
    street_shift = late_pressure - early_pressure

    return {
        "dbf_aggression_intensity": aggression_intensity.to_numpy(dtype=np.float64),
        "dbf_aggr_minus_fold": aggr_minus_fold.to_numpy(dtype=np.float64),
        "dbf_action_diversity_norm": np.asarray(action_diversity_norm, dtype=np.float64),
        "dbf_transition_complexity": np.asarray(transition_complexity, dtype=np.float64),
        "dbf_repetition_score": np.asarray(repetition_score, dtype=np.float64),
        "dbf_early_pressure": np.asarray(early_pressure, dtype=np.float64),
        "dbf_late_pressure": np.asarray(late_pressure, dtype=np.float64),
        "dbf_street_pressure_shift": np.asarray(street_shift, dtype=np.float64),
    }


def fit_dbf_quantile_bounds(
    df: pd.DataFrame,
    *,
    lo_q: float = 0.01,
    hi_q: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """
    Fit per-column winsor bounds on **training** rows only (raw DBF, pre-clip).

    Save with :func:`save_dbf_quantile_bounds` and pass the loaded mapping into
    :func:`compute_dbf_frame` / :func:`add_dbf_columns` for val/test/holdout so
    quantiles do not depend on the evaluation split.
    """
    raw = _compute_dbf_raw_arrays(df)
    out: dict[str, tuple[float, float]] = {}
    for name in DBF_COLUMN_NAMES:
        x = raw[name]
        if x.size == 0:
            out[name] = (0.0, 1.0)
            continue
        lo = float(np.nanquantile(x, lo_q))
        hi = float(np.nanquantile(x, hi_q))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            xf = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = float(np.min(xf)), float(np.max(xf))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -1.0, 1.0
        out[name] = (lo, hi)
    return out


def save_dbf_quantile_bounds(path: Path | str, bounds: Mapping[str, tuple[float, float]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: {"lo": float(lo), "hi": float(hi)} for k, (lo, hi) in bounds.items()}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dbf_quantile_bounds(path: Path | str) -> dict[str, tuple[float, float]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = (float(v[0]), float(v[1]))
        elif isinstance(v, dict) and "lo" in v and "hi" in v:
            out[k] = (float(v["lo"]), float(v["hi"]))
        else:
            raise ValueError(f"Invalid bounds entry for {k!r}: {v!r}")
    return out


def compute_dbf_frame(
    df: pd.DataFrame,
    *,
    quantile_bounds: Mapping[str, tuple[float, float]] | None = None,
    dbf_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Return a frame with ``dbf_*`` columns, row-aligned with ``df``.
    Missing inputs are treated as zeros for that sub-expression.

    If ``quantile_bounds`` is set, winsorization uses those fixed bounds per column
    (recommended for any split other than the one used to fit bounds). Only keys
    for ``dbf_columns`` (or all columns if omitted) are required.

    ``dbf_columns`` selects a subset of ``DBF_COLUMN_NAMES`` (e.g. after drift prune).
    """
    cols = normalize_dbf_column_subset(dbf_columns)
    raw = _compute_dbf_raw_arrays(df)
    if quantile_bounds is not None:
        missing = [c for c in cols if c not in quantile_bounds]
        if missing:
            raise ValueError(f"quantile_bounds missing columns: {missing}")
        robust = {
            k: _stabilize_feature(raw[k], clip_bounds=quantile_bounds[k]) for k in cols
        }
    else:
        robust = {k: _stabilize_feature(raw[k]) for k in cols}
    out = pd.DataFrame(robust, index=df.index)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def add_dbf_columns(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    quantile_bounds: Mapping[str, tuple[float, float]] | None = None,
    dbf_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append selected DBF columns; drops any existing ``dbf_*`` not in the selection."""
    cols = normalize_dbf_column_subset(dbf_columns)
    if inplace:
        base = df
    else:
        base = df.copy()
    for c in DBF_COLUMN_NAMES:
        if c in base.columns and c not in cols:
            base.drop(columns=[c], inplace=True)
    dbf = compute_dbf_frame(base, quantile_bounds=quantile_bounds, dbf_columns=cols)
    for c in cols:
        base[c] = dbf[c].to_numpy(dtype=np.float32)
    return base
