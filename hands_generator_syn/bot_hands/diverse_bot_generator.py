"""
Generate bot chunks using alternative BotProfile pools (same engine as generate_poker_data).

Use this when you want **wider behavioral diversity** without rewriting the hand
simulator. For integrating **other** algorithms (solvers, external bots), see
`workspace/docs/BOT_DIVERSITY.md`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hands_generator.bot_hands.extra_bot_profiles import PROFILE_POOLS, get_profile_pool
from hands_generator.data_generator import _default_bot_profiles, generate_bot_chunk


def merged_mixed_profiles() -> List[Any]:
    """Default mid-range + extended pools (dedupe by profile name, defaults first)."""
    seen = set()
    out = []
    for p in _default_bot_profiles():
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    for pool in PROFILE_POOLS.values():
        for p in pool:
            if p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


def generate_bot_chunk_diverse(
    size: int,
    pool: str = "mixed",
    *,
    reference_hands: Optional[List[Dict[str, Any]]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    pool:
      - "mixed" — `_default_bot_profiles()` + all `extra_bot_profiles` pools
      - "default" — same as `data_generator` default five profiles
      - any key from `PROFILE_POOLS` (e.g. "extremes", "sizing", "tilty")
    """
    if pool == "mixed":
        profiles = merged_mixed_profiles()
    elif pool == "default":
        profiles = _default_bot_profiles()
    else:
        profiles = get_profile_pool(pool)
    return generate_bot_chunk(size, profiles, reference_hands=reference_hands, seed=seed)
