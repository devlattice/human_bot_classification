#!/usr/bin/env python3
"""
Stream-merge *.jsonl hand logs from --input-dir into --output-dir.

- Line-by-line read (no full-file load).
- Deduplicate by ``chunk_hash``: first occurrence kept, later duplicates skipped.
- Lines without ``chunk_hash`` are written every time (logged once as warning per N).
- When the current output file would exceed --max-bytes (default 500 MiB), start
  ``merged_000002.jsonl``, etc. in the same directory.

Example::

    python3 workspace/ssl_data/merge.py \
        --input-dir workspace/ssl_data/json \
        --output-dir workspace/ssl_data/raw_data/source \
        --log-every 100

    #optional arguments    
        --max-bytes 500000000 \
        --output-stem merged \
        --no-dedupe 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Set, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "workspace" / "ssl_data" / "json"
DEFAULT_OUTPUT = REPO_ROOT / "workspace" / "ssl_data" / "raw_data" / "source"
DEFAULT_MAX_BYTES = 500 * 1024 * 1024
OUTPUT_STEM = "merged"


def _iter_jsonl_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir is not a directory: {input_dir}")
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl under {input_dir}")
    return files


class _RotatingWriter:
    """Append UTF-8 lines; rotate to merged_NNNNNN.jsonl when size exceeds max_bytes."""

    def __init__(self, out_dir: Path, max_bytes: int, stem: str = OUTPUT_STEM) -> None:
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.stem = stem
        self.part = 0
        self.bytes_in_part = 0
        self._f: Optional[TextIO] = None

    def _open_next(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        self.part += 1
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.stem}_{self.part:06d}.jsonl"
        path = self.out_dir / name
        self._f = path.open("w", encoding="utf-8", newline="\n")
        self.bytes_in_part = 0
        print(f"Writing {path}", file=sys.stderr)

    def write_line(self, line: str) -> None:
        """Write one logical line (no trailing newline in ``line``)."""
        payload = line + "\n"
        b = len(payload.encode("utf-8"))
        if self._f is None:
            self._open_next()
        assert self._f is not None
        if self.bytes_in_part > 0 and self.bytes_in_part + b > self.max_bytes:
            self._open_next()
            assert self._f is not None
        self._f.write(payload)
        self.bytes_in_part += b

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stream-merge JSONL hands; dedupe by chunk_hash; rotate large outputs."
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing *.jsonl (default: {DEFAULT_INPUT})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for merged_000001.jsonl, ... (default: {DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Start a new part when current file would exceed this size (default: {DEFAULT_MAX_BYTES})",
    )
    ap.add_argument(
        "--output-stem",
        default=OUTPUT_STEM,
        help=f'Base name for merged files, e.g. "{OUTPUT_STEM}" -> merged_000001.jsonl',
    )
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not skip duplicate chunk_hash (default: dedupe)",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Progress log every N lines read (0 = quiet)",
    )
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    seen: Set[str] = set()
    dedupe = not args.no_dedupe

    jsonl_files = _iter_jsonl_files(input_dir)
    writer = _RotatingWriter(output_dir, max_bytes=args.max_bytes, stem=args.output_stem)

    total_read = 0
    total_written = 0
    total_dup = 0
    total_bad = 0
    total_no_hash = 0
    warned_no_hash = False

    try:
        for src in jsonl_files:
            print(f"Reading {src}", file=sys.stderr)
            with src.open("r", encoding="utf-8") as inf:
                for lineno, raw in enumerate(inf, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    total_read += 1
                    if args.log_every and total_read % args.log_every == 0:
                        print(
                            f"  lines_read={total_read} written={total_written} "
                            f"dup={total_dup} bad={total_bad}",
                            file=sys.stderr,
                        )

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        total_bad += 1
                        print(f"  SKIP JSON error {src}:{lineno}", file=sys.stderr)
                        continue

                    h = obj.get("chunk_hash")
                    if dedupe and h is not None:
                        if not isinstance(h, str):
                            h = str(h)
                        if h in seen:
                            total_dup += 1
                            continue
                        seen.add(h)
                    elif dedupe and h is None:
                        total_no_hash += 1
                        if not warned_no_hash:
                            print(
                                "  Note: some lines lack chunk_hash; they are not deduped.",
                                file=sys.stderr,
                            )
                            warned_no_hash = True

                    writer.write_line(line)
                    total_written += 1
    finally:
        writer.close()

    print(
        f"Done: read={total_read} written={total_written} dup_skipped={total_dup} "
        f"bad_json={total_bad} no_chunk_hash={total_no_hash} unique_hashes={len(seen)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
