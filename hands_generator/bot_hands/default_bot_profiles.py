"""
Canonical default BotProfile list for chunk generation and CLI samples.

Single source of truth: `mixed_dataset_provider` bot_profile_mode "default",
`data_generator.generate_bot_chunk`, and `generate_poker_data` standalone entrypoints
all use these five profiles so behavior does not drift between modules.
"""

from __future__ import annotations

from typing import List

from hands_generator.bot_hands.sandbox_poker_bot import BotProfile


def default_bot_profiles() -> List[BotProfile]:
    return [
        BotProfile(
            name="balanced",
            tightness=0.58,
            aggression=0.58,
            bluff_freq=0.04,
            preflop_defend_bias=-0.10,
            postflop_continue_bias=-0.08,
            trap_frequency=-0.10,
        ),
        BotProfile(
            name="tight_aggressive",
            tightness=0.66,
            aggression=0.74,
            bluff_freq=0.04,
            preflop_defend_bias=-0.18,
            postflop_continue_bias=-0.14,
            trap_frequency=-0.06,
        ),
        BotProfile(
            name="loose_aggressive",
            tightness=0.48,
            aggression=0.74,
            bluff_freq=0.07,
            preflop_defend_bias=0.10,
            postflop_continue_bias=0.02,
            trap_frequency=0.00,
        ),
        BotProfile(
            name="tight_passive",
            tightness=0.64,
            aggression=0.42,
            bluff_freq=0.02,
            preflop_defend_bias=-0.24,
            postflop_continue_bias=-0.20,
            trap_frequency=-0.18,
        ),
        BotProfile(
            name="loose_passive",
            tightness=0.50,
            aggression=0.40,
            bluff_freq=0.04,
            preflop_defend_bias=-0.06,
            postflop_continue_bias=-0.10,
            trap_frequency=-0.12,
        ),
    ]
