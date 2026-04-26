"""
Drift / stability checks for ``dbf_*`` columns across train vs val / test.

Uses the same stabilized DBF values as downstream models (``compute_dbf_frame``
with train-fitted ``quantile_bounds``). Columns with large mean shift relative to
train dispersion, or near-constant train, are flagged unstable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .features import DBF_COLUMN_NAMES, compute_dbf_frame


def _mean_std(a: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(x)), float(np.nanstd(x))


def assess_dbf_column_stability(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_frames: dict[str, pd.DataFrame],
    quantile_bounds: Mapping[str, tuple[float, float]] | None,
    max_abs_mean_z: float = 3.0,
    min_train_std: float = 1e-5,
) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """
    Score each ``dbf_*`` column on mean shift vs train (after DBF transform).

    Returns ``(stable_columns, unstable_columns, detail)`` where stable/unstable
    partition ``DBF_COLUMN_NAMES``.
    """
    cap = float(max_abs_mean_z)
    eps = 1e-12
    min_std = float(min_train_std)

    tr = compute_dbf_frame(train_df, quantile_bounds=quantile_bounds)
    va = compute_dbf_frame(val_df, quantile_bounds=quantile_bounds)
    tests: dict[str, pd.DataFrame] = {}
    for name, df in test_frames.items():
        if len(df) == 0:
            continue
        tests[name] = compute_dbf_frame(df, quantile_bounds=quantile_bounds)

    per: dict[str, dict] = {}
    unstable: list[str] = []
    for col in DBF_COLUMN_NAMES:
        tcol = tr[col].to_numpy(dtype=np.float64)
        mu_t, sig_t = _mean_std(tcol)
        row: dict = {
            "train_mean": mu_t,
            "train_std": sig_t,
            "val_mean": float("nan"),
            "val_abs_z": float("nan"),
            "tests_max_abs_z": float("nan"),
            "reasons": [],
        }
        if not np.isfinite(sig_t) or sig_t < min_std:
            row["reasons"].append("low_train_std")
            unstable.append(col)
            per[col] = row
            continue

        vcol = va[col].to_numpy(dtype=np.float64)
        mu_v, _ = _mean_std(vcol)
        row["val_mean"] = mu_v
        zv = abs(mu_v - mu_t) / (sig_t + eps)
        row["val_abs_z"] = float(zv)

        max_tz = 0.0
        for tn, tdf in tests.items():
            m_s, _ = _mean_std(tdf[col].to_numpy(dtype=np.float64))
            z = abs(m_s - mu_t) / (sig_t + eps)
            max_tz = max(max_tz, float(z))
            row[f"test_{tn}_mean"] = float(m_s)
            row[f"test_{tn}_abs_z"] = float(z)
        row["tests_max_abs_z"] = float(max_tz)

        reasons: list[str] = []
        if zv > cap:
            reasons.append(f"val_mean_shift_z>{cap:g}")
        if max_tz > cap:
            reasons.append(f"test_mean_shift_z>{cap:g}")
        row["reasons"] = reasons
        if reasons:
            unstable.append(col)
        per[col] = row

    stable = tuple(c for c in DBF_COLUMN_NAMES if c not in unstable)
    detail = {
        "max_abs_mean_z": cap,
        "min_train_std": min_std,
        "per_column": per,
        "stable": list(stable),
        "unstable": unstable,
    }
    return stable, tuple(unstable), detail


def save_dbf_drift_report(path: Path | str, detail: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
