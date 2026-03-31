#!/usr/bin/env python3
"""
Semi-supervised / SSL data prep: read miner-style JSONL from a directory,
deduplicate by ``chunk_hash``, split records into low / medium / high
``risk_score`` buckets, write three JSON arrays.

Input lines are expected to be objects with at least:
``chunk_hash``, ``chunk``, ``risk_score`` (as produced by
``POKER44_MINER_LOG_CHUNK_NDJSON``).

Bucket rules (``risk_score`` in ``[0, 1]``):

- **low**: ``risk_score <= low_threshold``
- **high**: ``risk_score >= high_threshold``
- **medium**: ``low_threshold < risk_score < high_threshold``

Requires ``low_threshold < high_threshold``.

By default, existing ``score_*.json`` in the output directory are **merged**
(older ``chunk_hash`` entries are kept; new hashes from input are added).
Only those three filenames are written; **no other files** in the output
directory are removed. Use ``--replace-output`` to build solely from input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

OUT_NAMES = ("score_low.json", "score_medium.json", "score_high.json")

LOG_PREFIX = "[ssl_split]"


def log(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"{LOG_PREFIX} {msg}", file=sys.stderr)


def iter_input_records(
    input_dir: Path,
    *,
    quiet: bool,
    log_every: int,
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    """Yield (path, line_no, obj) from all ``*.jsonl`` and ``*.json`` files."""
    paths = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jsonl", ".json"}
    )
    log(f"found {len(paths)} input file(s) under {input_dir}", quiet=quiet)
    t0 = time.perf_counter()
    for path in paths:
        log(f"reading {path.name} …", quiet=quiet)
        n_ok = n_bad = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    n_bad += 1
                    log(
                        f"skip invalid JSON {path.name}:{line_no}: {e}",
                        quiet=quiet,
                    )
                    continue
                if not isinstance(obj, dict):
                    n_bad += 1
                    log(f"skip non-object {path.name}:{line_no}", quiet=quiet)
                    continue
                n_ok += 1
                if log_every > 0 and n_ok % log_every == 0:
                    log(
                        f"  {path.name}: {n_ok} records read "
                        f"({time.perf_counter() - t0:.1f}s elapsed)",
                        quiet=quiet,
                    )
                yield path, line_no, obj
        log(
            f"  {path.name}: done — {n_ok} ok, {n_bad} skipped lines "
            f"({time.perf_counter() - t0:.1f}s)",
            quiet=quiet,
        )


def _norm_hash(rec: dict[str, Any]) -> str | None:
    h = rec.get("chunk_hash")
    if h is None:
        return None
    s = str(h).strip()
    return s or None


def load_existing_outputs(
    out_dir: Path,
    *,
    quiet: bool,
) -> dict[str, dict[str, Any]]:
    """Load prior score_*.json arrays; first bucket wins per chunk_hash (low→med→high)."""
    merged: dict[str, dict[str, Any]] = {}
    for name in OUT_NAMES:
        path = out_dir / name
        if not path.is_file():
            continue
        t0 = time.perf_counter()
        try:
            with path.open("r", encoding="utf-8") as f:
                arr = json.load(f)
        except json.JSONDecodeError as e:
            log(f"warning: could not parse {path}: {e} (skipping merge for this file)", quiet=quiet)
            continue
        if not isinstance(arr, list):
            log(f"warning: {path} is not a JSON array (skipping)", quiet=quiet)
            continue
        added = 0
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            key = _norm_hash(rec)
            if key is None:
                continue
            if key not in merged:
                merged[key] = rec
                added += 1
        log(
            f"merged existing {name}: {len(arr)} rows, {added} new hashes "
            f"({time.perf_counter() - t0:.2f}s)",
            quiet=quiet,
        )
    log(f"existing total unique chunk_hash: {len(merged)}", quiet=quiet)
    return merged


def ingest_input(
    input_dir: Path,
    *,
    dedup: bool,
    merge_existing: bool,
    global_map: dict[str, dict[str, Any]],
    quiet: bool,
    log_every: int,
) -> tuple[int, int, int]:
    """
    Stream input into ``global_map``. Returns (n_added, n_skipped_dup_input,
    n_skipped_preserved_existing).
    """
    seen_input: set[str] = set()
    n_added = n_skip_in = n_skip_old = 0
    for _path, _line_no, rec in iter_input_records(
        input_dir, quiet=quiet, log_every=log_every
    ):
        key = _norm_hash(rec)
        if key is None:
            log("skip record without chunk_hash", quiet=quiet)
            continue
        if dedup:
            if key in seen_input:
                n_skip_in += 1
                continue
            seen_input.add(key)
        if merge_existing and key in global_map:
            n_skip_old += 1
            continue
        global_map[key] = rec
        n_added += 1
    return n_added, n_skip_in, n_skip_old


def bucket_all(
    global_map: dict[str, dict[str, Any]],
    *,
    low_t: float,
    high_t: float,
    quiet: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    low_l: list[dict] = []
    mid_l: list[dict] = []
    high_l: list[dict] = []
    n_bad = 0
    for rec in global_map.values():
        try:
            s = float(rec["risk_score"])
        except (KeyError, TypeError, ValueError):
            n_bad += 1
            continue
        if s <= low_t:
            low_l.append(rec)
        elif s >= high_t:
            high_l.append(rec)
        else:
            mid_l.append(rec)
    if n_bad:
        log(f"skipped {n_bad} records without numeric risk_score when bucketing", quiet=quiet)
    return low_l, mid_l, high_l


def write_json_array(
    path: Path,
    rows: list[dict],
    *,
    pretty: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(rows, f, indent=2, ensure_ascii=True)
            f.write("\n")
        else:
            json.dump(rows, f, separators=(",", ":"), ensure_ascii=True)
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split SSL JSONL records by risk_score with optional chunk_hash dedup."
    )
    parser.add_argument(
        "--input-json-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "json",
        help="Directory containing *.jsonl / *.json (one JSON object per line).",
    )
    parser.add_argument(
        "--output-json-dir",
        type=Path,
        required=True,
        help="Directory for score_low.json, score_medium.json, score_high.json",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.33,
        help="Scores <= this go to score_low.json (default: 0.33).",
    )
    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.67,
        help="Scores >= this go to score_high.json (default: 0.67).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication by chunk_hash within the input stream (default: dedup on).",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Ignore existing score_*.json; output only from input (default: merge with existing).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (slower, larger files). Default: compact one-line arrays.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal logging.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Log progress every N input records per file (0=disable). Default: 100000.",
    )
    args = parser.parse_args()
    low_t = float(args.low_score_threshold)
    high_t = float(args.high_score_threshold)
    if low_t >= high_t:
        log(
            "Error: require --low-score-threshold < --high-score-threshold",
            quiet=False,
        )
        return 1

    input_dir: Path = args.input_json_dir
    if not input_dir.is_dir():
        log(f"Error: input dir not found: {input_dir}", quiet=False)
        return 1

    out_dir: Path = args.output_json_dir
    dedup = not args.no_dedup
    merge_existing = not args.replace_output
    quiet = args.quiet
    log_every = max(0, int(args.log_every))

    t_all = time.perf_counter()
    if merge_existing:
        global_map = load_existing_outputs(out_dir, quiet=quiet)
    else:
        global_map = {}
        log("replace-output: starting from empty maps (no merge)", quiet=quiet)

    n_added, n_skip_in, n_skip_old = ingest_input(
        input_dir,
        dedup=dedup,
        merge_existing=merge_existing,
        global_map=global_map,
        quiet=quiet,
        log_every=log_every,
    )
    log(
        f"ingest: +{n_added} from input, "
        f"skipped {n_skip_in} duplicate-in-input, "
        f"skipped {n_skip_old} already in existing output",
        quiet=quiet,
    )

    low_rows, mid_rows, high_rows = bucket_all(
        global_map, low_t=low_t, high_t=high_t, quiet=quiet
    )

    log("writing outputs (compact JSON by default; other files in dir untouched) …", quiet=quiet)
    write_json_array(out_dir / OUT_NAMES[0], low_rows, pretty=args.pretty)
    write_json_array(out_dir / OUT_NAMES[1], mid_rows, pretty=args.pretty)
    write_json_array(out_dir / OUT_NAMES[2], high_rows, pretty=args.pretty)

    elapsed = time.perf_counter() - t_all
    print(
        f"{LOG_PREFIX} done in {elapsed:.2f}s | "
        f"input={input_dir} | dedup={'on' if dedup else 'off'} | "
        f"merge_existing={'on' if merge_existing else 'off'} | "
        f"thresholds low<={low_t}, high>={high_t} | "
        f"unique chunk_hash={len(global_map)} | "
        f"buckets low={len(low_rows)} medium={len(mid_rows)} high={len(high_rows)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
