"""
Runtime DBF (Distilled Behavior Features) for miner chunk rows.

Training pipeline for ``*_with_dbf`` parquets applies ``transform_meta`` to base chunk
columns, then appends ``dbf_*`` via ``workspace/teacher/dbf/features.py`` using
train-fitted quantile bounds. The miner mirrors that order: aggregate → transform_meta
→ DBF merge (this module) → optional SSL → ``predict_proba``.

Imports the teacher ``dbf`` package lazily (requires ``workspace/teacher`` on ``sys.path``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

_teacher_path_prepared = False


def _ensure_teacher_dbf_import(repo_root: Path) -> None:
    global _teacher_path_prepared
    if _teacher_path_prepared:
        return
    teacher = (repo_root / "workspace" / "teacher").resolve()
    if not teacher.is_dir():
        raise ImportError(f"Teacher DBF path not found: {teacher}")
    p = str(teacher)
    if p not in sys.path:
        sys.path.insert(0, p)
    _teacher_path_prepared = True


def load_dbf_quantile_bounds_file(repo_root: Path, path: Path) -> dict[str, tuple[float, float]]:
    _ensure_teacher_dbf_import(repo_root)
    from dbf.features import load_dbf_quantile_bounds

    return load_dbf_quantile_bounds(path)


def compute_dbf_row_values(
    repo_root: Path,
    row: Mapping[str, Any],
    quantile_bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    """
    Return ``dbf_*`` values for one row dict (already transform_meta-aligned).

    Column subset follows ``dbf.py``: only DBF names present in ``quantile_bounds``,
    in canonical ``DBF_COLUMN_NAMES`` order.
    """
    _ensure_teacher_dbf_import(repo_root)
    from dbf.features import DBF_COLUMN_NAMES, compute_dbf_frame

    cols = tuple(c for c in DBF_COLUMN_NAMES if c in quantile_bounds)
    if not cols:
        return {}
    df = pd.DataFrame([dict(row)])
    fr = compute_dbf_frame(df, quantile_bounds=quantile_bounds, dbf_columns=cols)
    return {c: float(fr[c].iloc[0]) for c in cols}
