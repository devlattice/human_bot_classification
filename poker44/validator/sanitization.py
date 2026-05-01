"""Miner-visible hand normalization (compat alias for ``payload_view``)."""

from __future__ import annotations

from poker44.validator.payload_view import prepare_hand_for_miner as sanitize_hand_for_miner

__all__ = ["sanitize_hand_for_miner"]
