"""May-8-conditioned ``PokerHandGenerator`` factory (Phase 1 + Phase 2).

Phase 1:
  - Lock sb/bb/max_seats per matched candidate (no zenodo stake resampling)
  - BB-scaled stacks (no absolute 8–12 chip hero stacks)
  - Reference distribution from May-8 gold bot hands when available

Phase 2:
  - ``passive_may8_mode`` on ``BotProfile`` → check/call-heavy policy in ``SandboxPokerBot``

Phase 3:
  - ``passive_raise_rate`` + ``passive_pot_build_mult`` → micro-raises and larger pots
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hands_generator.bot_hands.generate_poker_data import (
    PokerHandGenerator,
    load_gold_bot_reference_hands,
)
from hands_generator.bot_hands.sandbox_poker_bot import BotProfile

REPO = Path(__file__).resolve().parents[3]
DEFAULT_GOLD_DIR = REPO / "workspace" / "dataset" / "source" / "gold_dataset"

_ref_cache: list[dict[str, Any]] | None = None


def get_may8_reference_hands(
    gold_dir: Path | None = None,
    *,
    max_hands: int = 8000,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    global _ref_cache
    if _ref_cache is not None and not refresh:
        return _ref_cache
    path = gold_dir or DEFAULT_GOLD_DIR
    _ref_cache = load_gold_bot_reference_hands(path, date_substr="2026-05-08", max_hands=max_hands)
    return _ref_cache


def make_may8_generator(
    candidate: dict[str, Any],
    *,
    seed: int,
    gold_dir: Path | None = None,
    use_reference: bool = True,
) -> PokerHandGenerator:
    """Build generator with table config locked to LHS match candidate."""
    ref = get_may8_reference_hands(gold_dir) if use_reference else None
    max_seats = int(candidate.get("max_seats", 6))
    target = int(candidate.get("target_players", max_seats))
    return PokerHandGenerator(
        sb=float(candidate["sb"]),
        bb=float(candidate["bb"]),
        max_seats=max_seats,
        rake_rate=0.05,
        reference_hands=ref if ref else None,
        seed=seed,
        lock_table_config=True,
        target_players=target,
    )


def bot_profile_from_candidate(
    profile_kwargs: dict[str, Any],
    name: str,
    *,
    passive_may8: bool = False,
) -> BotProfile:
    _passive_keys = (
        "passive_may8_mode", "passive_check_bias", "passive_bet_scale",
        "passive_raise_rate", "passive_pot_build_mult",
    )
    kw = {k: v for k, v in profile_kwargs.items() if k not in _passive_keys}
    if not passive_may8:
        return BotProfile(name=name, **kw)
    agg = float(kw.get("aggression", 0.15))
    small_frac = float(kw.get("bet_pot_fraction_small", 0.25))
    med_frac = float(kw.get("bet_pot_fraction_medium", 0.40))
    return BotProfile(
        name=name,
        passive_may8_mode=True,
        passive_check_bias=max(0.68, min(0.86, 0.80 - 0.20 * agg)),
        passive_bet_scale=max(0.40, min(0.72, 0.42 + 0.55 * small_frac)),
        passive_raise_rate=max(0.08, min(0.22, 0.10 + 0.35 * agg)),
        passive_pot_build_mult=max(1.35, min(2.25, 1.20 + 1.10 * med_frac)),
        **kw,
    )
