"""Resolve paths to the public human hand corpus (same files as mixed_dataset_provider)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

HUMAN_HANDS_DIR = Path(__file__).resolve().parent


def resolve_default_human_corpus_path() -> Optional[Path]:
    """Prefer `poker_hands_combined.json.gz`, then `.json`; return ``None`` if neither exists."""
    for name in ("poker_hands_combined.json.gz", "poker_hands_combined.json"):
        p = HUMAN_HANDS_DIR / name
        if p.exists():
            return p
    return None
