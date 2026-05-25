#!/usr/bin/env python3
"""Count total rows/hands from JSON or JSONL input.

Usage:
  python workspace/utils/get_total_row.py \
    --input-json workspace/dataset/source/data/zenodo_v3/poker_hands_zenodo_train.json \
    --progress-every 10000
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def _is_jsonl(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".jsonl") or n.endswith(".jsonl.gz")


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _count_jsonl(path: Path, *, progress_every: int = 0) -> int:
    count = 0
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                count += 1
                if progress_every > 0 and count % progress_every == 0:
                    print(f"[progress] rows={count}", flush=True)
    return count


def _count_json(path: Path, *, progress_every: int = 0) -> int:
    # Streaming count for top-level JSON array to avoid OOM on huge files.
    decoder = json.JSONDecoder()
    with _open_text(path) as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if first != "[":
            # Small non-array JSON fallback.
            f.seek(0)
            obj = json.load(f)
            if isinstance(obj, list):
                return len(obj)
            if isinstance(obj, dict):
                if isinstance(obj.get("hands"), list):
                    return len(obj["hands"])
                if isinstance(obj.get("data"), list):
                    return len(obj["data"])
            raise SystemExit("Unsupported JSON structure: expected list or dict with 'hands'/'data' list.")

        count = 0
        buf = ""
        eof = False
        done = False
        while not done:
            if not eof:
                chunk = f.read(1 << 20)  # 1MB
                if chunk:
                    buf += chunk
                else:
                    eof = True
            i = 0
            n = len(buf)
            while True:
                while i < n and buf[i].isspace():
                    i += 1
                if i >= n:
                    break
                if buf[i] == "]":
                    done = True
                    i += 1
                    break
                if buf[i] == ",":
                    i += 1
                    continue
                try:
                    _, j = decoder.raw_decode(buf, i)
                except json.JSONDecodeError:
                    break
                count += 1
                if progress_every > 0 and count % progress_every == 0:
                    print(f"[progress] rows={count}", flush=True)
                i = j
            if i > 0:
                buf = buf[i:]
            if eof and not done:
                raise SystemExit("Malformed JSON array or truncated file.")
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Count total rows from JSON/JSONL file.")
    parser.add_argument("--input-json", type=Path, required=True, help="Input .json/.jsonl (optionally .gz)")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N rows while counting (0 disables).",
    )
    args = parser.parse_args()

    p = args.input_json.expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"Input not found: {p}")

    if _is_jsonl(p):
        total = _count_jsonl(p, progress_every=max(0, int(args.progress_every)))
    else:
        total = _count_json(p, progress_every=max(0, int(args.progress_every)))
    print(total)


if __name__ == "__main__":
    main()
