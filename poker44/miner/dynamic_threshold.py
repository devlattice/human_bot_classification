"""Production threshold policy: fixed fallback + bimodal gap (ScoreMonitor-style).

Used by the miner per validator request and by offline production eval scripts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ThresholdConfig:
    fixed_fallback: float = 0.5
    static_selected: float | None = None  # bundle retrain_summary selected_threshold
    clamp_min: float = 0.10
    clamp_max: float = 0.45
    min_scores: int = 20
    min_cluster_frac: float = 0.05
    gap_ratio: float = 5.0
    min_gap_abs: float = 0.01
    rolling_window: int = 500
    prefer_batch: bool = True  # use current request batch if large enough


@dataclass
class ThresholdDecision:
    threshold: float
    mode: str  # "gap_batch" | "gap_rolling" | "static" | "fixed"
    meta: dict[str, Any] = field(default_factory=dict)


class DynamicThresholdPolicy:
    """Choose decision threshold from rolling / batch score distribution."""

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()
        self._rolling: deque[float] = deque(maxlen=max(50, self.config.rolling_window))
        self._last: ThresholdDecision | None = None

    def observe(self, scores: list[float]) -> None:
        for s in scores:
            self._rolling.append(float(s))

    def _clamp(self, t: float) -> float:
        return max(self.config.clamp_min, min(self.config.clamp_max, float(t)))

    def _bimodal_gap_threshold(self, arr: np.ndarray) -> tuple[float | None, dict[str, Any]]:
        if len(arr) < self.config.min_scores:
            return None, {"reason": "too_few_scores", "n": int(len(arr))}

        sorted_s = np.sort(arr)
        gaps = np.diff(sorted_s)
        if len(gaps) == 0:
            return None, {"reason": "no_gaps"}

        gap_idx = int(np.argmax(gaps))
        max_gap = float(gaps[gap_idx])
        median_gap = float(np.median(gaps)) if len(gaps) else 0.0
        is_bimodal = max_gap > self.config.gap_ratio * max(median_gap, 1e-6) and max_gap > self.config.min_gap_abs

        meta: dict[str, Any] = {
            "n": int(len(arr)),
            "max_gap": round(max_gap, 5),
            "median_gap": round(median_gap, 5),
            "bimodal": bool(is_bimodal),
            "gap_idx": gap_idx,
        }

        if not is_bimodal:
            return None, meta

        boundary = float((sorted_s[gap_idx] + sorted_s[gap_idx + 1]) / 2.0)
        low = arr[arr < boundary]
        high = arr[arr >= boundary]
        min_each = max(1, int(len(arr) * self.config.min_cluster_frac))
        meta.update({
            "boundary": round(boundary, 5),
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "human_peak": round(float(np.median(low)), 5) if len(low) else None,
            "bot_peak": round(float(np.median(high)), 5) if len(high) else None,
        })

        if len(low) < min_each or len(high) < min_each:
            meta["reason"] = "cluster_too_small"
            return None, meta

        return boundary, meta

    def decide(self, batch_scores: list[float] | None = None) -> ThresholdDecision:
        cfg = self.config
        batch_arr: np.ndarray | None = None
        if batch_scores and len(batch_scores) >= cfg.min_scores:
            batch_arr = np.asarray(batch_scores, dtype=np.float64)

        if cfg.prefer_batch and batch_arr is not None:
            t_gap, meta = self._bimodal_gap_threshold(batch_arr)
            if t_gap is not None:
                dec = ThresholdDecision(
                    threshold=self._clamp(t_gap),
                    mode="gap_batch",
                    meta=meta,
                )
                self._last = dec
                return dec

        if len(self._rolling) >= cfg.min_scores:
            roll_arr = np.asarray(list(self._rolling), dtype=np.float64)
            t_gap, meta = self._bimodal_gap_threshold(roll_arr)
            if t_gap is not None:
                dec = ThresholdDecision(
                    threshold=self._clamp(t_gap),
                    mode="gap_rolling",
                    meta={**meta, "rolling_n": len(self._rolling)},
                )
                self._last = dec
                return dec

        if cfg.static_selected is not None:
            dec = ThresholdDecision(
                threshold=self._clamp(cfg.static_selected),
                mode="static",
                meta={"static_selected": cfg.static_selected},
            )
            self._last = dec
            return dec

        dec = ThresholdDecision(
            threshold=self._clamp(cfg.fixed_fallback),
            mode="fixed",
            meta={"fixed_fallback": cfg.fixed_fallback},
        )
        self._last = dec
        return dec

    def decide_and_observe(self, batch_scores: list[float]) -> ThresholdDecision:
        dec = self.decide(batch_scores)
        self.observe(batch_scores)
        return dec
