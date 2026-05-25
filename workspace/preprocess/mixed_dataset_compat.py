"""
Workspace-local helpers for `build_dataset.py` so we do **not** patch `hands_generator/`.

The subnet clone updates `hands_generator` often; `MixedDatasetConfig` fields may lag or
differ from this workspace CLI. We keep richer `--bot-*` flags here and only pass kwargs
the subnet dataclass actually accepts.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List

from hands_generator.mixed_dataset_provider import MixedDatasetConfig

# Modes our CLI documents; if subnet `MixedDatasetConfig` has no `bot_profile_mode`,
# those values are accepted on the CLI then dropped with a notice when building kwargs.
_BOT_PROFILE_MODE_CHOICES: List[str] = [
    "default",
    "mixed",
    "rotate",
    "extremes",
    "sizing",
    "tilty",
]


def all_bot_profile_modes() -> List[str]:
    """Returned modes for `argparse` choices; subnet may still ignore unknown fields."""
    return list(_BOT_PROFILE_MODE_CHOICES)


def filter_mixed_dataset_config_kwargs(cfg_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys that the current subnet `MixedDatasetConfig` does not define."""
    allowed = {f.name for f in fields(MixedDatasetConfig)}
    filtered = {k: v for k, v in cfg_kw.items() if k in allowed}
    dropped = sorted(set(cfg_kw) - set(filtered))
    if cfg_kw and dropped:
        print(
            "[build_dataset] Subnet `MixedDatasetConfig` has no field(s) "
            f"{dropped}; they are ignored. Use a subnet revision that adds them for full effect.",
            flush=True,
        )
    return filtered
