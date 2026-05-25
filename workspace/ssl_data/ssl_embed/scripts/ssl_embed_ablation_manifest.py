#!/usr/bin/env python3
"""
Helpers for ssl_embed_ablation_manifest.json (frozen SSL + LGBM hyperparameters).

Written by tune_ssl_lgbm_optuna.py after nested search; consumed by run_ssl_lgbm_ablation.sh
(via shell-exports) and workspace/model/scripts/lgbm_2.py (via --hparams-json, lgbm section).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def load_manifest(path: Path) -> dict[str, Any]:
    p = path.expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        print(
            f"[warn] manifest schema_version={data.get('schema_version')!r} "
            f"expected {SCHEMA_VERSION}",
            file=sys.stderr,
        )
    return data


def shell_exports(manifest_path: Path) -> str:
    """Print bash `export VAR=...` lines (safe-quoted) for run_ssl_lgbm_ablation.sh."""
    m = load_manifest(manifest_path)
    ssl = m.get("ssl") or {}
    lines: list[str] = []

    def _exp(name: str, val: Any) -> None:
        s = "" if val is None else str(val)
        lines.append(f"export {name}={shlex.quote(s)}")

    _exp("SSL_MASK_RATIO", ssl.get("mask_ratio", "0.30"))
    _exp("SSL_MASK_MODE", ssl.get("mask_mode", "random"))
    _exp("SSL_MASK_MIXED_ALPHA", ssl.get("mask_mixed_alpha", "0.3"))
    _exp("SSL_EMBED_DIM", ssl.get("embed_dim", "32"))
    _exp("SSL_HIDDEN_DIM", ssl.get("hidden_dim", "96"))
    _exp("SSL_MAX_ITER", ssl.get("max_iter", "80"))
    _exp("SSL_SEED", m.get("ssl_seed", ssl.get("seed", "42")))

    mw = (m.get("paths") or {}).get("mask_weight_json") or ""
    _exp("SSL_MASK_WEIGHT_JSON", mw if mw else "")

    _exp("LGBM_DEVICE", m.get("lgbm_device", "cpu"))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="ssl_embed ablation manifest utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_shell = sub.add_parser("shell-exports", help="Print bash export lines for ablation script.")
    p_shell.add_argument("--manifest", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "shell-exports":
        sys.stdout.write(shell_exports(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
