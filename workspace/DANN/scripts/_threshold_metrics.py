"""Threshold sweep on labeled validation (human=0, bot=1).

Predict bot when p_bot >= t. Human FPR = (# humans predicted bot) / (# humans).
Bot recall = (# bots predicted bot) / (# bots).

Selection rule (``bot_recall_at_human_fpr_cap``):
  Among thresholds with human_fpr <= cap, maximize bot_recall; break ties by
  threshold closest to ``tie_ref``.
  If no threshold meets the cap, minimize human_fpr, then maximize bot_recall,
  then tie-break by closeness to ``tie_ref``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def metrics_at_threshold(y_true_01: np.ndarray, p_bot: np.ndarray, t: float) -> Tuple[float, float, float]:
    """Returns (human_fpr, bot_recall, accuracy_at_t)."""
    y = (np.asarray(y_true_01, dtype=np.float64).reshape(-1) >= 0.5).astype(np.int64)
    p = np.asarray(p_bot, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    pred = (p >= float(t)).astype(np.int64)

    n_h = int((y == 0).sum())
    n_b = int((y == 1).sum())
    fp_h = int(((y == 0) & (pred == 1)).sum())
    tp_b = int(((y == 1) & (pred == 1)).sum())

    human_fpr = float(fp_h) / float(n_h) if n_h > 0 else 0.0
    bot_recall = float(tp_b) / float(n_b) if n_b > 0 else 0.0
    acc = float((pred == y).sum()) / float(len(y)) if len(y) > 0 else 0.0
    return human_fpr, bot_recall, acc


def select_threshold_bot_recall_under_human_fpr_cap(
    y_true_01: np.ndarray,
    p_bot: np.ndarray,
    human_fpr_cap: float,
    grid_size: int,
    tie_ref: float,
) -> Dict[str, Any]:
    """Grid search t in [0, 1]; return best row + flags."""
    cap = float(human_fpr_cap)
    g = max(2, int(grid_size))
    thresholds = np.linspace(0.0, 1.0, g, dtype=np.float64)
    tie_ref = float(tie_ref)

    best_feas: Dict[str, Any] | None = None
    best_feas_key: Tuple[float, float, float] | None = None  # (br, -hf, -dist)

    best_infeas: Dict[str, Any] | None = None
    best_infeas_key: Tuple[float, float, float, float] | None = None  # (-hf, br, -dist)

    for t in thresholds:
        hf, br, acc = metrics_at_threshold(y_true_01, p_bot, float(t))
        feasible = hf <= cap + 1e-12
        dist = abs(float(t) - tie_ref)
        row = {
            "threshold": float(t),
            "human_fpr": hf,
            "bot_recall": br,
            "accuracy_at_t": acc,
            "feasible": bool(feasible),
            "human_fpr_cap": cap,
            "threshold_grid_size": int(g),
        }
        if feasible:
            k = (br, -hf, -dist)
            if best_feas_key is None or k > best_feas_key:
                best_feas_key = k
                best_feas = row
        else:
            k2 = (-hf, br, -dist)
            if best_infeas_key is None or k2 > best_infeas_key:
                best_infeas_key = k2
                best_infeas = row

    if best_feas is not None:
        chosen = dict(best_feas)
    elif best_infeas is not None:
        chosen = dict(best_infeas)
    else:
        chosen = {
            "threshold": float(tie_ref),
            "human_fpr": 0.0,
            "bot_recall": 0.0,
            "accuracy_at_t": 0.0,
            "feasible": True,
            "human_fpr_cap": cap,
            "threshold_grid_size": int(g),
        }

    chosen["selection_rule"] = (
        "max_bot_recall_subject_to_human_fpr_cap_else_min_human_fpr_then_max_bot_recall"
    )
    return chosen


def epoch_rank_tuple(
    selection_metric: str,
    val_acc_05: float,
    sweep: Dict[str, Any] | None,
    domain_confusion_loss: float | None = None,
    *,
    multi_objective: Dict[str, Any] | None = None,
) -> Tuple[float, ...]:
    """Lexicographic tuple: larger is better (compare with >)."""
    if selection_metric == "val_acc":
        return (val_acc_05,)
    if selection_metric == "bot_recall_at_human_fpr_cap":
        if sweep is None:
            return (float("-inf"),)
        feas = 1.0 if sweep.get("feasible") else 0.0
        br = float(sweep.get("bot_recall", 0.0))
        hf = float(sweep.get("human_fpr", 1.0))
        return (feas, br, -hf)
    if selection_metric == "bot_recall_at_human_fpr_cap_then_domain_confusion":
        if sweep is None:
            return (float("-inf"),)
        feas = 1.0 if sweep.get("feasible") else 0.0
        br = float(sweep.get("bot_recall", 0.0))
        hf = float(sweep.get("human_fpr", 1.0))
        dloss = float("-inf") if domain_confusion_loss is None else float(domain_confusion_loss)
        return (feas, br, -hf, dloss)
    if selection_metric == "multi_objective_generalization":
        if multi_objective is None:
            return (float("-inf"),)
        # Lexicographic priority:
        # 1) enforce FPR feasibility across all validation domains
        # 2) maximize worst-domain bot recall (robustness)
        # 3) maximize average bot recall
        # 4) minimize average human FPR
        # 5) prefer stronger domain confusion tie-break
        all_feasible = 1.0 if bool(multi_objective.get("all_feasible", False)) else 0.0
        worst_br = float(multi_objective.get("worst_bot_recall", 0.0))
        mean_br = float(multi_objective.get("mean_bot_recall", 0.0))
        mean_hf = float(multi_objective.get("mean_human_fpr", 1.0))
        dloss = float("-inf") if domain_confusion_loss is None else float(domain_confusion_loss)
        return (all_feasible, worst_br, mean_br, -mean_hf, dloss)
    raise ValueError(f"unknown selection_metric: {selection_metric!r}")
