"""
train_qc: row-level contract, invariants, row-health thresholds fit on **gold train** only,
duplicate fingerprints, tiering, and artifacts under ``<bundle>/qc/<run_id>/``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
# v2: default invariants assume **robust-scaled** chunk features (e.g. total_dataset parquets),
# not raw [0,1] ratios. Only finite checks + sample_weight sanity.
RULE_PACK = "train_qc_v2_robust_features"


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"label", "sample_weight"}
    return [c for c in df.columns if c not in skip]


def _invariant_flags(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Boolean Series per row: non-finite feature values or bad ``sample_weight``."""
    n = len(df)
    X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float64, copy=True)
    bad = np.zeros(n, dtype=bool)
    bad |= np.any(np.isposinf(X) | np.isneginf(X), axis=1)
    if "sample_weight" in df.columns:
        w = pd.to_numeric(df["sample_weight"], errors="coerce").to_numpy(dtype=np.float64)
        bad |= ~np.isfinite(w) | (w < 0.0)
    return pd.Series(bad, index=df.index, dtype=bool)


def _row_health_metrics(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float64, copy=True)
    nan_frac = np.mean(~np.isfinite(X), axis=1)
    X0 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    active = np.sum(np.abs(X0) > 1e-8, axis=1)
    l2 = np.linalg.norm(X0, axis=1)
    return nan_frac, active.astype(np.float64), l2


def _dup_cluster_ids(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """(cluster_id int, flag_label_conflict bool) — conflict only meaningful on labeled rows."""
    X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float64, copy=True)
    Xr = np.round(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 4)
    # pandas hashable row keys
    keys = [hashlib.md5(row.tobytes(), usedforsecurity=False).hexdigest()[:16] for row in Xr]
    ser = pd.Series(keys, index=df.index, dtype=str)
    # cluster id: map key to small int
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    cid = ser.map(lambda k: uniq[k]).astype(np.int64)

    conflict = pd.Series(False, index=df.index)
    if "label" in df.columns:
        for _, sub in df.groupby(cid, sort=False):
            if len(sub) <= 1:
                continue
            labs = pd.to_numeric(sub["label"], errors="coerce").dropna().unique()
            if len(labs) > 1:
                conflict.loc[sub.index] = True
    return cid, conflict


def _tier(
    inv: pd.Series,
    nan_frac: np.ndarray,
    l2: np.ndarray,
    dup_conflict: pd.Series,
    thr: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    tier = np.full(len(inv), "clean", dtype=object)
    reasons: list[str] = [""] * len(inv)

    def add(i: int, t: str, r: str) -> None:
        cur = tier[i]
        if cur == "hard":
            return
        if t == "hard":
            tier[i] = "hard"
            reasons[i] = r
            return
        if cur == "soft":
            reasons[i] = f"{reasons[i]}|{r}" if reasons[i] else r
            return
        if t == "soft":
            tier[i] = "soft"
            reasons[i] = r

    hi_nan = float(thr["nan_frac_soft_above"])
    hard_nan = float(thr["nan_frac_hard_above"])
    l2_soft_below = float(thr["l2_soft_below"])

    for i in range(len(inv)):
        if inv.iloc[i]:
            add(i, "hard", "invariant")
        elif nan_frac[i] >= hard_nan:
            add(i, "hard", "nan_frac")
        elif nan_frac[i] > hi_nan:
            add(i, "soft", "nan_frac")
        if l2[i] < l2_soft_below:
            add(i, "soft", "low_l2")
        if dup_conflict.iloc[i]:
            add(i, "soft", "dup_label_conflict")

    return pd.Series(tier, dtype=str), pd.Series(reasons, dtype=str)


def _fit_thresholds_gold(nan_frac_g: np.ndarray, l2_g: np.ndarray) -> dict[str, float]:
    """Quantiles on gold train only; conservative defaults if degenerate."""
    if len(nan_frac_g) == 0:
        return {
            "nan_frac_soft_above": 0.05,
            "nan_frac_hard_above": 0.95,
            "l2_soft_below": 1e-6,
        }
    q99 = float(np.quantile(nan_frac_g, 0.99))
    q01_l2 = float(np.quantile(l2_g[l2_g > 0], 0.01)) if np.any(l2_g > 0) else 1e-8
    hi_soft = max(q99 * 3.0, 0.02)
    hi_soft = min(hi_soft, 0.45)
    hard = max(0.9, hi_soft + 0.1)
    hard = min(hard, 0.999)
    l2_soft = max(q01_l2 * 0.25, 1e-8)
    return {
        "nan_frac_soft_above": hi_soft,
        "nan_frac_hard_above": hard,
        "l2_soft_below": l2_soft,
    }


def run_train_qc_on_bundle(
    bundle_dir: Path,
    *,
    run_id: str | None = None,
    max_gold_hard_frac: float = 0.02,
    max_val_hard_frac: float = 0.05,
) -> dict[str, Any]:
    """
    Read ``manifest.json`` under ``bundle_dir``, QC ``out_train`` / ``out_val``,
    write ``qc/<run_id>/`` artifacts and merge QC fields into ``manifest.json``.
    """
    bundle_dir = bundle_dir.expanduser().resolve()
    man_path = bundle_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {man_path}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    train_path = Path(manifest["out_train"]).expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train parquet: {train_path}")

    n_gold = int(manifest.get("n_gold", 0))
    if n_gold <= 0:
        raise ValueError("manifest n_gold must be positive")

    train_df = pd.read_parquet(train_path)
    feature_cols = _feature_columns(train_df)
    if not feature_cols:
        raise ValueError("No feature columns found (expected label + features + optional sample_weight)")

    nan_frac, active, l2 = _row_health_metrics(train_df, feature_cols)
    inv = _invariant_flags(train_df, feature_cols)
    dup_id, dup_conflict = _dup_cluster_ids(train_df, feature_cols)

    gold_mask = np.arange(len(train_df)) < n_gold
    thr = _fit_thresholds_gold(nan_frac[gold_mask], l2[gold_mask])
    tier_tr, reason_tr = _tier(inv, nan_frac, l2, dup_conflict, thr)

    val_path = manifest.get("out_val")
    val_df: pd.DataFrame | None = None
    tier_va = reason_va = nan_frac_va = active_va = l2_va = inv_va = dup_id_va = dup_conflict_va = None
    if val_path:
        vp = Path(val_path).expanduser().resolve()
        if vp.is_file():
            val_df = pd.read_parquet(vp)
            v_feats = _feature_columns(val_df)
            if set(v_feats) != set(feature_cols):
                raise ValueError(
                    f"val feature set mismatch vs train: "
                    f"{sorted(set(feature_cols) ^ set(v_feats))[:24]}"
                )
            nan_frac_va, active_va, l2_va = _row_health_metrics(val_df, feature_cols)
            inv_va = _invariant_flags(val_df, feature_cols)
            dup_id_va, dup_conflict_va = _dup_cluster_ids(val_df, feature_cols)
            tier_va, reason_va = _tier(inv_va, nan_frac_va, l2_va, dup_conflict_va, thr)

    rid = run_id or _utc_run_id()
    qc_root = bundle_dir / "qc" / rid
    qc_root.mkdir(parents=True, exist_ok=True)

    qc_config = {
        "schema_version": SCHEMA_VERSION,
        "rule_pack": RULE_PACK,
        "bundle_dir": str(bundle_dir),
        "n_gold_train": n_gold,
        "n_feature_cols": len(feature_cols),
        "feature_col_hash": hashlib.sha256(
            json.dumps(feature_cols, sort_keys=True).encode()
        ).hexdigest()[:16],
    }
    (qc_root / "qc_config.json").write_text(json.dumps(qc_config, indent=2), encoding="utf-8")
    (qc_root / "thresholds_fit.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")

    flags_train = pd.DataFrame(
        {
            "row_index": np.arange(len(train_df), dtype=np.int64),
            "qc_tier": tier_tr.to_numpy(),
            "qc_reason": reason_tr.to_numpy(),
            "qc_nan_frac": nan_frac,
            "qc_active_features": active,
            "qc_l2": l2,
            "qc_flag_invariant": inv.to_numpy(dtype=np.int8),
            "qc_dup_cluster": dup_id.to_numpy(),
            "qc_dup_label_conflict": dup_conflict.to_numpy(dtype=np.int8),
            "qc_is_gold_train": gold_mask.astype(np.int8),
        }
    )
    flags_train.to_parquet(qc_root / "train.flags.parquet", index=False)

    if val_df is not None and tier_va is not None:
        flags_val = pd.DataFrame(
            {
                "row_index": np.arange(len(val_df), dtype=np.int64),
                "qc_tier": tier_va.to_numpy(),
                "qc_reason": reason_va.to_numpy(),
                "qc_nan_frac": nan_frac_va,
                "qc_active_features": active_va,
                "qc_l2": l2_va,
                "qc_flag_invariant": inv_va.to_numpy(dtype=np.int8),
                "qc_dup_cluster": dup_id_va.to_numpy(),
                "qc_dup_label_conflict": dup_conflict_va.to_numpy(dtype=np.int8),
                "qc_is_gold_train": np.zeros(len(val_df), dtype=np.int8),
            }
        )
        flags_val.to_parquet(qc_root / "val.flags.parquet", index=False)

    gold_hard_frac = float((tier_tr.iloc[:n_gold] == "hard").mean()) if n_gold else 0.0
    train_hard_frac = float((tier_tr == "hard").mean())
    soft_frac = float((tier_tr == "soft").mean())

    summary: dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)) if val_df is not None else 0,
        "tier_counts_train": tier_tr.value_counts().to_dict(),
        "tier_counts_val": tier_va.value_counts().to_dict() if tier_va is not None else {},
        "gold_train_hard_frac": gold_hard_frac,
        "train_hard_frac": train_hard_frac,
        "train_soft_frac": soft_frac,
        "n_invariant_train": int(inv.sum()),
        "n_invariant_val": int(inv_va.sum()) if inv_va is not None else 0,
    }
    (qc_root / "qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    gate_fail = gold_hard_frac > max_gold_hard_frac
    if val_df is not None and inv_va is not None:
        val_hard_frac = float((tier_va == "hard").mean())
        summary["val_hard_frac"] = val_hard_frac
        if val_hard_frac > max_val_hard_frac:
            gate_fail = True
    gate = {"status": "fail" if gate_fail else "pass", "gold_hard_frac": gold_hard_frac}
    (qc_root / "gate_result.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    manifest["train_qc"] = {
        "run_id": rid,
        "dir": str(qc_root.relative_to(bundle_dir)).replace("\\", "/"),
        "gate": gate["status"],
        "rule_pack": RULE_PACK,
    }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"run_id": rid, "qc_root": str(qc_root), "gate": gate["status"], "summary": summary}
