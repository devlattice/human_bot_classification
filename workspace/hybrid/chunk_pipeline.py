"""Chunk feature extraction for the hybrid training stack.

Must stay aligned with ``poker44.validator.payload_view`` (miner-visible hands)
and ``neurons/miner.py`` (``skip_sanitize=True`` on validator-delivered chunks).

Two paths:

* **Raw / offline hands** (gold benchmark JSON, zenodo, hand generator output):
  run ``sanitize_hand_for_miner`` (``build_miner_payload_hand``) before features.
  Action leaks are stripped, amounts are bucketed, and only a small action window
  is kept — matching live evaluation.

* **Miner payload chunks** (logged validator requests, synapse ``chunks``):
  already sanitized at send time; do **not** sanitize again.
"""

from __future__ import annotations

from typing import Any, Dict, List

from poker44.validator.chunk_features import (
    aggregate_chunk_from_hands,
    miner_servable_feature_names,
)

# Bump when ``payload_view`` / feature schema changes; retrain.sh compares this stamp.
FEATURE_PIPELINE_VERSION = "payload-view-action-leak-tighten-2026-05"


def aggregate_chunk_from_raw_hands(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Offline training data: full hands → miner-visible → chunk features."""
    return aggregate_chunk_from_hands(hands, skip_sanitize=False)


def aggregate_chunk_from_miner_payload(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Live miner path: hands already passed through ``build_miner_payload_hand``."""
    return aggregate_chunk_from_hands(hands, skip_sanitize=True)


__all__ = [
    "FEATURE_PIPELINE_VERSION",
    "aggregate_chunk_from_raw_hands",
    "aggregate_chunk_from_miner_payload",
    "aggregate_chunk_from_hands",
    "miner_servable_feature_names",
]
