"""
Chunk-level feature aggregation for miner-visible hands.

Used by ``workspace/datasets/preprocess_lightgbm`` (training) and ``neurons/miner``
(reference heuristic) so both apply the same pipeline:

  raw hand JSON → ``sanitize_hand_for_miner`` → per-hand numeric features
  → mean / std / max across hands in the chunk → one feature vector per chunk.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

from poker44.validator.sanitization import sanitize_hand_for_miner

MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold", "all_in")


def hand_features_miner_view(sanitized: Dict[str, Any]) -> Dict[str, float]:
    """Numeric features from one miner-visible hand (after sanitization)."""
    actions = sanitized.get("actions") or []
    players = sanitized.get("players") or []
    streets = sanitized.get("streets") or []

    types = Counter(str(a.get("action_type") or "") for a in actions)
    meaningful = max(
        1,
        sum(types.get(k, 0) for k in MEANINGFUL_ACTIONS),
    )

    norm_amts = [float(a.get("normalized_amount_bb") or 0.0) for a in actions]
    pot_after = [float(a.get("pot_after") or 0.0) for a in actions]

    n_streets = float(len(streets))
    n_players = float(len(players))
    n_players_i = int(n_players)
    # Sanitized streets are present but board cards are hidden; use street-count buckets
    # as a robust structural proxy (preflop-only vs flop/turn/river reach rates).
    end_preflop = 1.0 if n_streets <= 0 else 0.0
    end_flop = 1.0 if n_streets == 1 else 0.0
    end_turn = 1.0 if n_streets == 2 else 0.0
    end_river = 1.0 if n_streets >= 3 else 0.0
    p2 = 1.0 if n_players_i <= 2 else 0.0
    p3 = 1.0 if n_players_i == 3 else 0.0
    p4 = 1.0 if n_players_i == 4 else 0.0
    p5 = 1.0 if n_players_i == 5 else 0.0
    p6p = 1.0 if n_players_i >= 6 else 0.0

    return {
        "n_players": float(len(players)),
        "n_streets": float(len(streets)),
        "n_actions_slot": float(len(actions)),
        "call_ratio": types.get("call", 0) / meaningful,
        "check_ratio": types.get("check", 0) / meaningful,
        "fold_ratio": types.get("fold", 0) / meaningful,
        "raise_ratio": types.get("raise", 0) / meaningful,
        "bet_ratio": types.get("bet", 0) / meaningful,
        "all_in_ratio": types.get("all_in", 0) / meaningful,
        "other_ratio": types.get("other", 0) / meaningful,
        "mean_norm_bb": float(np.mean(norm_amts)) if norm_amts else 0.0,
        "std_norm_bb": float(np.std(norm_amts)) if norm_amts else 0.0,
        "max_norm_bb": float(np.max(norm_amts)) if norm_amts else 0.0,
        "mean_pot_after": float(np.mean(pot_after)) if pot_after else 0.0,
        "std_pot_after": float(np.std(pot_after)) if pot_after else 0.0,
        "end_preflop": end_preflop,
        "end_flop": end_flop,
        "end_turn": end_turn,
        "end_river": end_river,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6p": p6p,
        "stack_mean": float(
            np.mean([float(p.get("starting_stack") or 0.0) for p in players])
        )
        if players
        else 0.0,
        "stack_std": float(
            np.std([float(p.get("starting_stack") or 0.0) for p in players])
        )
        if len(players) > 1
        else 0.0,
    }


def aggregate_chunk_from_hands(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """One row: mean / std / max of per-hand miner-view features (same as training)."""
    if not hands:
        return {}

    per = [hand_features_miner_view(sanitize_hand_for_miner(h)) for h in hands]
    keys = per[0].keys()
    out: Dict[str, float] = {"chunk_n_hands": float(len(hands))}
    for k in keys:
        vals = [row[k] for row in per]
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr))
        out[f"{k}_max"] = float(np.max(arr))
    return out


def miner_servable_feature_names() -> tuple[str, ...]:
    """
    Ordered column names for one chunk row produced by :func:`aggregate_chunk_from_hands`
    (excluding ``label``). Same schema a live miner builds from ``DetectionSynapse.chunks``.

    Use this to validate training Parquet columns and avoid zero-filled extras at inference.
    """
    ref_hands: List[Dict[str, Any]] = [{"actions": [], "players": [], "streets": []}]
    row = aggregate_chunk_from_hands(ref_hands)
    if not row:
        raise RuntimeError("aggregate_chunk_from_hands returned empty for reference chunk")
    return tuple(row.keys())
