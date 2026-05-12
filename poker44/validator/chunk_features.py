"""
Chunk-level feature aggregation for miner-visible hands.

Used by ``workspace/datasets/preprocess_lightgbm`` (training) and ``neurons/miner``
(reference heuristic) so both apply the same pipeline:

  raw hand JSON → ``sanitize_hand_for_miner`` → per-hand numeric features
  → mean / std / max (+ selected p10/p50/p90) across hands in the chunk
  → small set of deterministic contrast features
  → one feature vector per chunk.

Feature isolation: ``"other"`` actions are counted as a standalone signal but
excluded from all behavioral feature computations (amounts, pot, entropy, etc.)
so that removal of ``"other"`` by a future validator version does not shift any
other feature.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from poker44.validator.sanitization import sanitize_hand_for_miner

MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold", "all_in")
AGGRESSIVE_ACTIONS = ("bet", "raise", "all_in")
PASSIVE_ACTIONS = ("call", "check")
EPS = 1e-6

# Percentile summaries are always emitted for these per-hand keys.
PERCENTILE_KEYS = (
    "bet_ratio",
    "call_ratio",
    "check_ratio",
    "fold_ratio",
    "raise_ratio",
    "mean_norm_bb",
    "mean_pot_after",
    "aggression_factor",
    "action_entropy",
    "bet_size_mean",
)


def _action_entropy(types: Counter, meaningful: int) -> float:
    """Shannon entropy over meaningful action type distribution."""
    if meaningful <= 0:
        return 0.0
    entropy = 0.0
    for k in MEANINGFUL_ACTIONS:
        c = types.get(k, 0)
        if c > 0:
            p = c / meaningful
            entropy -= p * math.log2(p)
    return entropy


def hand_features_miner_view(sanitized: Dict[str, Any]) -> Dict[str, float]:
    """Numeric features from one miner-visible hand (after sanitization).

    All behavioral features are computed on *meaningful actions only* (excluding
    ``"other"`` and blind postings).  The ``"other"`` signal is captured
    separately as ``other_ratio`` / ``other_count``.
    """
    actions = sanitized.get("actions") or []
    players = sanitized.get("players") or []
    streets = sanitized.get("streets") or []

    # Separate meaningful actions from "other" / unknown
    all_types = Counter(str(a.get("action_type") or "") for a in actions)
    meaningful_acts = [
        a for a in actions
        if str(a.get("action_type") or "") in MEANINGFUL_ACTIONS
    ]
    types = Counter(str(a.get("action_type") or "") for a in meaningful_acts)
    meaningful = max(1, len(meaningful_acts))

    # Amounts/pot computed ONLY on meaningful actions (immune to "other")
    norm_amts = [float(a.get("normalized_amount_bb") or 0.0) for a in meaningful_acts]
    pot_after = [float(a.get("pot_after") or 0.0) for a in meaningful_acts]
    nonzero_amts = [x for x in norm_amts if x > 0.0]

    n_streets = float(len(streets))
    n_players = float(len(players))
    n_players_i = int(n_players)

    end_preflop = 1.0 if n_streets <= 0 else 0.0
    end_flop = 1.0 if n_streets == 1 else 0.0
    end_turn = 1.0 if n_streets == 2 else 0.0
    end_river = 1.0 if n_streets >= 3 else 0.0
    p2 = 1.0 if n_players_i <= 2 else 0.0
    p3 = 1.0 if n_players_i == 3 else 0.0
    p4 = 1.0 if n_players_i == 4 else 0.0
    p5 = 1.0 if n_players_i == 5 else 0.0
    p6p = 1.0 if n_players_i >= 6 else 0.0

    # --- Layer 1: isolated "other" signal (standalone, survives as 0 if removed) ---
    other_count = all_types.get("other", 0)
    total_all = max(1, len(actions))
    other_ratio = other_count / total_all

    # --- Layer 2: robust behavioral features (meaningful actions only) ---

    # Action type ratios
    call_ratio = types.get("call", 0) / meaningful
    check_ratio = types.get("check", 0) / meaningful
    fold_ratio = types.get("fold", 0) / meaningful
    raise_ratio = types.get("raise", 0) / meaningful
    bet_ratio = types.get("bet", 0) / meaningful
    all_in_ratio = types.get("all_in", 0) / meaningful

    # Aggression factor: (bet+raise+all_in) / (call+check+1)
    n_aggressive = sum(types.get(k, 0) for k in AGGRESSIVE_ACTIONS)
    n_passive = sum(types.get(k, 0) for k in PASSIVE_ACTIONS)
    aggression_factor = n_aggressive / (n_passive + 1.0)

    # Action entropy (diversity of action choices)
    action_entropy = _action_entropy(types, meaningful)

    # Bet sizing features (only non-zero amounts)
    bet_size_mean = float(np.mean(nonzero_amts)) if nonzero_amts else 0.0
    bet_size_std = float(np.std(nonzero_amts)) if len(nonzero_amts) > 1 else 0.0
    bet_size_max = float(np.max(nonzero_amts)) if nonzero_amts else 0.0

    # Pot dynamics
    pot_growth = 0.0
    if len(pot_after) >= 2:
        pot_growth = (pot_after[-1] - pot_after[0]) / max(1.0, float(len(pot_after)))

    # Sequential patterns (on meaningful actions only)
    m_types_seq = [str(a.get("action_type") or "") for a in meaningful_acts]
    max_consecutive = 0.0
    if m_types_seq:
        run = 1
        for i in range(1, len(m_types_seq)):
            if m_types_seq[i] == m_types_seq[i - 1]:
                run += 1
            else:
                if run > max_consecutive:
                    max_consecutive = float(run)
                run = 1
        max_consecutive = max(max_consecutive, float(run))

    # Unique actors participating in meaningful actions
    actors = set(a.get("actor_seat") for a in meaningful_acts if a.get("actor_seat"))
    unique_actors_ratio = len(actors) / max(1.0, n_players) if n_players > 0 else 0.0

    # Fold position: normalized position of fold actions within the sequence
    fold_positions = [
        i / max(1, meaningful - 1)
        for i, a in enumerate(meaningful_acts)
        if str(a.get("action_type") or "") == "fold"
    ]
    fold_position_mean = float(np.mean(fold_positions)) if fold_positions else 0.0

    # Preflop action density (how much of the action happens preflop)
    preflop_count = sum(
        1 for a in meaningful_acts if str(a.get("street") or "") == "preflop"
    )
    preflop_action_density = preflop_count / meaningful

    return {
        # Structure
        "n_players": n_players,
        "n_streets": n_streets,
        "n_actions": float(meaningful),
        # Action type ratios (meaningful only)
        "call_ratio": call_ratio,
        "check_ratio": check_ratio,
        "fold_ratio": fold_ratio,
        "raise_ratio": raise_ratio,
        "bet_ratio": bet_ratio,
        "all_in_ratio": all_in_ratio,
        # Isolated "other" signal (Layer 1)
        "other_ratio": other_ratio,
        # Amount / pot features (meaningful only)
        "mean_norm_bb": float(np.mean(norm_amts)) if norm_amts else 0.0,
        "std_norm_bb": float(np.std(norm_amts)) if norm_amts else 0.0,
        "max_norm_bb": float(np.max(norm_amts)) if norm_amts else 0.0,
        "mean_pot_after": float(np.mean(pot_after)) if pot_after else 0.0,
        "std_pot_after": float(np.std(pot_after)) if pot_after else 0.0,
        # Bet sizing (non-zero amounts only)
        "bet_size_mean": bet_size_mean,
        "bet_size_std": bet_size_std,
        "bet_size_max": bet_size_max,
        # Street structure
        "end_preflop": end_preflop,
        "end_flop": end_flop,
        "end_turn": end_turn,
        "end_river": end_river,
        # Player count buckets
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6p": p6p,
        # Stack features
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
        # New behavioral features (all on meaningful actions, immune to "other")
        "aggression_factor": aggression_factor,
        "action_entropy": action_entropy,
        "pot_growth": pot_growth,
        "max_consecutive": max_consecutive,
        "unique_actors_ratio": unique_actors_ratio,
        "fold_position_mean": fold_position_mean,
        "preflop_action_density": preflop_action_density,
    }


def aggregate_chunk_from_hands(
    hands: List[Dict[str, Any]],
    *,
    skip_sanitize: bool = False,
) -> Dict[str, float]:
    """One row: aggregated + derived chunk features (same as training).

    When *skip_sanitize* is True the hands are assumed to be already sanitized
    (e.g. validator-sent chunks received by miners at runtime).  This preserves
    action types like ``"other"`` that would otherwise be destroyed by the local
    sanitization pass.
    """
    if not hands:
        return {}

    if skip_sanitize:
        per = [hand_features_miner_view(h) for h in hands]
    else:
        per = [hand_features_miner_view(sanitize_hand_for_miner(h)) for h in hands]
    keys = per[0].keys()
    out: Dict[str, float] = {"chunk_n_hands": float(len(hands))}
    for k in keys:
        vals = [row[k] for row in per]
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_std"] = float(np.std(arr))
        out[f"{k}_max"] = float(np.max(arr))
        if k in PERCENTILE_KEYS:
            out[f"{k}_p10"] = float(np.quantile(arr, 0.10))
            out[f"{k}_p50"] = float(np.quantile(arr, 0.50))
            out[f"{k}_p90"] = float(np.quantile(arr, 0.90))

    # Deterministic contrast features from stable aggregated primitives.
    out["raise_minus_call_mean"] = float(out["raise_ratio_mean"] - out["call_ratio_mean"])
    out["bet_minus_fold_mean"] = float(out["bet_ratio_mean"] - out["fold_ratio_mean"])
    out["late_minus_early_mean"] = float(
        (out["end_turn_mean"] + out["end_river_mean"])
        - (out["end_preflop_mean"] + out["end_flop_mean"])
    )
    out["raise_std_over_check_std"] = float(
        out["raise_ratio_std"] / (abs(out["check_ratio_std"]) + EPS)
    )
    out["pot_after_over_stack_mean"] = float(
        out["mean_pot_after_mean"] / (abs(out["stack_mean_mean"]) + EPS)
    )
    # Behavioral contrast features
    out["aggression_std_over_mean"] = float(
        out["aggression_factor_std"] / (abs(out["aggression_factor_mean"]) + EPS)
    )
    out["entropy_range"] = float(out["action_entropy_max"] - out.get("action_entropy_mean", 0.0))
    out["bet_size_cv"] = float(
        out["bet_size_mean_std"] / (abs(out["bet_size_mean_mean"]) + EPS)
    )
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
