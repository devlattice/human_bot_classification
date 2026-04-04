#!/usr/bin/env python3
"""Shim: delegates to canonical ``human_bot_validator/prepare_weak_ssl_dataset.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CANONICAL = (
    Path(__file__).resolve().parents[2]
    / "usl_hdbscan"
    / "human_bot_validator"
    / "prepare_weak_ssl_dataset.py"
)


def _load_main():
    spec = importlib.util.spec_from_file_location("_prepare_weak_ssl_canonical", _CANONICAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_CANONICAL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


if __name__ == "__main__":
    if not _CANONICAL.is_file():
        print(f"[prepare_weak_ssl] error: missing canonical script {_CANONICAL}", file=sys.stderr)
        raise SystemExit(1)
    main = _load_main()
    raise SystemExit(main())
