#!/usr/bin/env python3
"""
Normalize real-distribution hands to the same top-level schema as
``hands_generator/human_hands/poker_hands_combined.json`` (Poker44 V0 hand JSON).

``merged.json`` here is newline-delimited JSON (one hand object per line): same
nested shape as the combined corpus but **no** ``label`` field. This script adds
``label`` with JSON ``null`` for unknown supervision (pandas / numpy load that
as NaN). Existing string labels (``human``, ``bot``, ``ai``) are kept.

Hands are **deduplicated** by a stable hash of
``metadata`` + ``players`` + ``streets`` + ``actions`` + ``outcome`` (``label``
is ignored so the same hand is not counted twice). First occurrence wins; use
``--no-deduplicate`` to disable.

Examples (from repo root):

  PYTHONPATH=. python workspace/real_distribution/transfer.py \\
      --input workspace/real_distribution/processed/merged.json \\
      --output workspace/real_distribution/processed/merged_labeled.json

  # Keep NDJSON on output (one hand per line, no surrounding array):
  PYTHONPATH=. python workspace/real_distribution/transfer.py -i merged.json -o out.jsonl --format ndjson
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_CONTENT_KEYS: Tuple[str, ...] = ("metadata", "players", "streets", "actions", "outcome")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poker44.core.hand_json import V0_JSON_HAND

_CANONICAL_TOP_KEYS: List[str] = list(V0_JSON_HAND.keys())


def normalize_hand(
    hand: Dict[str, Any],
    *,
    unknown_label: Any = None,
) -> Dict[str, Any]:
    """
    Return a hand dict with the same key set/order as ``V0_JSON_HAND``.

    ``unknown_label`` is used when ``label`` is missing or already null (default
    ``None`` → JSON ``null``). Set to ``float("nan")`` only if you also write
    with ``allow_nan=True`` (non-standard JSON).
    """
    if "label" not in hand:
        label: Any = unknown_label
    else:
        label = hand["label"]

    ordered: Dict[str, Any] = {
        "metadata": hand["metadata"],
        "players": hand["players"],
        "streets": hand["streets"],
        "actions": hand["actions"],
        "outcome": hand["outcome"],
        "label": label,
    }
    extra = set(hand.keys()) - set(_CANONICAL_TOP_KEYS)
    if extra:
        raise ValueError(f"Unexpected top-level keys (not in V0 schema): {sorted(extra)}")
    return ordered


def hand_content_fingerprint(hand: Dict[str, Any]) -> str:
    """
    Stable identity for deduplication: full hand history without ``label``.
    Same logical hand matches even if top-level key order differs.
    """
    core = {k: hand[k] for k in _CONTENT_KEYS}
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _peek_first_non_space_char(path: Path) -> Optional[str]:
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return None
            if not ch.isspace():
                return ch


def iter_hands(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield hand dicts from a JSON array file or NDJSON (one object per line)."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    first = _peek_first_non_space_char(path)
    if first is None:
        return

    if first == "[":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON value is not an array.")
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item {i} is not an object.")
            yield item
        return

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected an object per line.")
            yield item


def write_hands_ndjson(
    path: Path,
    hands: Iterator[Dict[str, Any]],
    *,
    allow_nan: bool = False,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for h in hands:
            out.write(json.dumps(h, ensure_ascii=False, allow_nan=allow_nan))
            out.write("\n")
            n += 1
    return n


def write_hands_json_array(
    path: Path,
    hands: Iterator[Dict[str, Any]],
    *,
    indent: Optional[int] = None,
    allow_nan: bool = False,
) -> int:
    """Write a top-level JSON array (same container style as poker_hands_combined.json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = ", "
    if indent is not None:
        sep = ","
    n = 0
    with path.open("w", encoding="utf-8") as out:
        out.write("[")
        if indent is not None:
            out.write("\n")
        first = True
        for h in hands:
            if not first:
                out.write(sep)
                if indent is not None:
                    out.write("\n")
            first = False
            chunk = json.dumps(
                h,
                ensure_ascii=False,
                indent=indent,
                allow_nan=allow_nan,
            )
            if indent is not None:
                pref = " " * indent
                chunk = "\n".join(pref + ln if ln else ln for ln in chunk.split("\n"))
            out.write(chunk)
            n += 1
        if indent is not None:
            out.write("\n")
        out.write("]")
    return n


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_in = here / "processed" / "merged.json"
    default_out = here / "processed" / "merged_labeled.json"
    p.add_argument("--input", "-i", type=Path, default=default_in, help="Source JSON array or NDJSON.")
    p.add_argument("--output", "-o", type=Path, default=default_out, help="Destination path.")
    p.add_argument(
        "--format",
        "-f",
        choices=("array", "ndjson"),
        default="array",
        help="Output: JSON array (like poker_hands_combined.json) or one JSON object per line.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=None,
        metavar="N",
        help="Pretty-print with this indent (array format only). Default: compact.",
    )
    p.add_argument(
        "--allow-json-nan",
        action="store_true",
        help="Allow IEEE NaN in output (non-standard JSON; use with --unknown-nan).",
    )
    p.add_argument(
        "--unknown-nan",
        action="store_true",
        help="Use float NaN as unknown label instead of null (combine with --allow-json-nan).",
    )
    p.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Emit every input row even when hand history matches a prior row.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    unknown: Any = float("nan") if args.unknown_nan else None
    if args.unknown_nan and not args.allow_json_nan:
        print(
            "[transfer] warning: --unknown-nan without --allow-json-nan may raise; "
            "enabling allow_nan for output.",
            file=sys.stderr,
        )
        allow_nan = True
    else:
        allow_nan = bool(args.allow_json_nan)

    dedup = not args.no_deduplicate
    seen: set[str] = set()
    n_in = 0
    n_dup = 0

    def normalized_stream() -> Iterator[Dict[str, Any]]:
        nonlocal n_in, n_dup
        for hand in iter_hands(args.input):
            n_in += 1
            if dedup:
                fp = hand_content_fingerprint(hand)
                if fp in seen:
                    n_dup += 1
                    continue
                seen.add(fp)
            yield normalize_hand(hand, unknown_label=unknown)

    if args.format == "ndjson":
        n = write_hands_ndjson(args.output, normalized_stream(), allow_nan=allow_nan)
    else:
        n = write_hands_json_array(
            args.output,
            normalized_stream(),
            indent=args.indent,
            allow_nan=allow_nan,
        )
    if dedup:
        print(
            f"[transfer] input={n_in} unique={n} duplicates_dropped={n_dup} -> {args.output}",
            flush=True,
        )
    else:
        print(f"[transfer] wrote {n} hands (dedup off) -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
