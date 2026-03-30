"""
Extra BotProfile pools for dataset diversity.

The subnet's default profiles (`bot_hands/default_bot_profiles.py`, re-exported from `data_generator._default_bot_profiles`) cover a small
mid-range band. These profiles push **tighter/looser**, **more/less aggressive**,
**different sizing**, and **tilt-like** variance so LightGBM sees more than one
behavior manifold—still using the same `SandboxPokerBot` + `PokerHandGenerator` stack.

This is **not** a second poker engine; for that see `workspace/docs/BOT_DIVERSITY.md`.
"""

from __future__ import annotations

from typing import Dict, List

from hands_generator.bot_hands.sandbox_poker_bot import BotProfile

# --- Single-style families (use alone for “one bot type” chunks) ---

PROFILE_POOL_EXTREMES: List[BotProfile] = [
    BotProfile(
        name="extreme_nit",
        tightness=0.82,
        aggression=0.32,
        bluff_freq=0.01,
        max_risk_fraction_of_stack=0.12,
        tilt_factor=0.0,
        bet_pot_fraction_small=0.22,
        bet_pot_fraction_medium=0.40,
        bet_pot_fraction_large=0.55,
    ),
    BotProfile(
        name="mani_aggressive",
        tightness=0.28,
        aggression=0.92,
        bluff_freq=0.16,
        max_risk_fraction_of_stack=0.30,
        tilt_factor=0.05,
        bet_pot_fraction_small=0.45,
        bet_pot_fraction_medium=0.72,
        bet_pot_fraction_large=1.05,
    ),
    BotProfile(
        name="calling_station",
        tightness=0.38,
        aggression=0.28,
        bluff_freq=0.02,
        max_risk_fraction_of_stack=0.22,
        tilt_factor=0.0,
        bet_pot_fraction_small=0.28,
        bet_pot_fraction_medium=0.45,
        bet_pot_fraction_large=0.62,
    ),
    BotProfile(
        name="hyper_aggro_smallball",
        tightness=0.48,
        aggression=0.88,
        bluff_freq=0.12,
        max_risk_fraction_of_stack=0.15,
        tilt_factor=0.08,
        bet_pot_fraction_small=0.25,
        bet_pot_fraction_medium=0.42,
        bet_pot_fraction_large=0.58,
    ),
]

PROFILE_POOL_SIZING_FREAKS: List[BotProfile] = [
    BotProfile(
        name="tiny_bets",
        tightness=0.52,
        aggression=0.58,
        bluff_freq=0.06,
        max_risk_fraction_of_stack=0.14,
        bet_pot_fraction_small=0.15,
        bet_pot_fraction_medium=0.28,
        bet_pot_fraction_large=0.38,
    ),
    BotProfile(
        name="pot_splash",
        tightness=0.50,
        aggression=0.72,
        bluff_freq=0.09,
        max_risk_fraction_of_stack=0.28,
        bet_pot_fraction_small=0.55,
        bet_pot_fraction_medium=0.88,
        bet_pot_fraction_large=1.20,
    ),
]

PROFILE_POOL_TILTY: List[BotProfile] = [
    BotProfile(
        name="tilt_prone",
        tightness=0.55,
        aggression=0.62,
        bluff_freq=0.10,
        max_risk_fraction_of_stack=0.20,
        tilt_factor=0.28,
        bet_pot_fraction_small=0.38,
        bet_pot_fraction_medium=0.62,
        bet_pot_fraction_large=0.92,
    ),
    BotProfile(
        name="steam_raise",
        tightness=0.42,
        aggression=0.85,
        bluff_freq=0.11,
        max_risk_fraction_of_stack=0.26,
        tilt_factor=0.32,
        bet_pot_fraction_small=0.40,
        bet_pot_fraction_medium=0.68,
        bet_pot_fraction_large=0.98,
    ),
]

# Named registry for CLI / scripts
PROFILE_POOLS: Dict[str, List[BotProfile]] = {
    "extremes": PROFILE_POOL_EXTREMES,
    "sizing": PROFILE_POOL_SIZING_FREAKS,
    "tilty": PROFILE_POOL_TILTY,
}


def list_profile_pool_names() -> List[str]:
    return sorted(PROFILE_POOLS.keys())


def get_profile_pool(name: str) -> List[BotProfile]:
    if name not in PROFILE_POOLS:
        raise KeyError(f"Unknown profile pool {name!r}. Available: {list_profile_pool_names()}")
    return PROFILE_POOLS[name]
