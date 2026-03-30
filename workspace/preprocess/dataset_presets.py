"""
Defaults for building mixed human/bot training data.

- **training** — larger chunk count for offline model fitting (default preset).
- **validator-parity** — aligns with `neurons/validator.py` env defaults (`POKER44_*`).

Per-hand style is always `sanitize_hand_for_miner` (see `preprocess_lightgbm.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

PresetName = Literal["training", "validator-parity", "training-merged"]

# Validator neuron defaults (neurons/validator.py) — distribution only, not private human path.
VALIDATOR_ENV_LIKE: Dict[str, Any] = {
    "chunk_count": 40,
    "min_hands_per_chunk": 60,
    "max_hands_per_chunk": 120,
    "human_ratio": 0.5,
}

# Recommended for miner offline training (more rows; same hand range as validator).
TRAINING_LIKE: Dict[str, Any] = {
    "chunk_count": 120,
    "min_hands_per_chunk": 60,
    "max_hands_per_chunk": 120,
    "human_ratio": 0.5,
}

# Large offline run when human_json has ~100k+ hands (e.g. merged combined + Zenodo).
# Output row count ≈ chunk_count (each row = one chunk, many hands per chunk).
# With human_ratio=0.5, human hands requested ≈ (chunk_count/2) * ~90 ≈ chunk_count * 45.
# chunk_count≈2920 → ~130k human hands (matches large merged pools). Bot generation is slow.
TRAINING_MERGED_LIKE: Dict[str, Any] = {
    "chunk_count": 2920,
    "min_hands_per_chunk": 60,
    "max_hands_per_chunk": 120,
    "human_ratio": 0.5,
}

PRESETS: Dict[PresetName, Dict[str, Any]] = {
    "training": TRAINING_LIKE,
    "validator-parity": VALIDATOR_ENV_LIKE,
    "training-merged": TRAINING_MERGED_LIKE,
}

DEFAULT_BOT_PROFILE_MODE: str = "mixed"


def default_output_dir(repo_root: Path) -> Path:
    # Under repo root (user-owned) — avoid workspace/datasets when that tree is root-owned (e.g. Docker).
    return repo_root / "workspace" / "datasets" / "lgbm"
