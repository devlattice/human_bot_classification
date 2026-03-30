#!/usr/bin/env python3
"""Miner request log processing (separate from the miner process).

**Merge** (``--merge``, recommended): finds ``miner_requests_<n>.jsonl`` and
``miner_requests-<n>.jsonl`` in ``--directory``. If **only one** such file exists, **does
nothing** (that file is the live miner log). If **two or more** exist, assumes the **largest**
tag is the current log; picks the **smallest** tag among the rest, appends **unique hands**
to ``merged.jsonl`` (default: ``processed/merged.jsonl`` next to this script): **one JSON hand
object per line** (same shape as each element inside ``chunks``, no ``hand_hash`` wrapper).
Deduplication uses an internal content hash. Legacy ``merged.jsonl`` lines with
``{"hand_hash", "hand"}`` are still read for dedupe. Then **deletes** that picked file. No
full-file copy. Use ``--merge`` on the CLI (not the default in-place mode).

Optional ``--delete-empty-logs`` removes **0-byte** files matching ``--glob`` under
``--directory`` after each cycle (merge or in-place), e.g. old corrupted empty logs. You must pass **``--merge``** on the command line for this mode (it is not
the default). Use **``--delete-empty-logs``** to remove stale **0-byte** ``miner_requests*.jsonl``
files in ``--directory`` after each cycle.

**In-place (default without ``--merge``)**: same tagging rules — only files with tag **less
than the current max**; **never runs** when there is only one tagged file. Rewrites each
eligible file in place with deduped chunk or hand lines.

Slim miner logs (``{"chunks":...}`` only, no ``chunk_hashes``) **cannot** use chunk dedupe;
use ``--unique-by hand`` or **``--merge``** (hands → ``merged.jsonl`` and **deletes** the
source tag file). In-place **never** replaces a file with an empty body when all lines were
skipped (unless ``--in-place-delete-if-no-output``).

Dedupe uses **chunk_hashes** + **chunks** (or **hands** with ``--unique-by hand``).
Merge always dedupes **hands** and is what fills ``merged.jsonl``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GLOB = "miner_requests*.jsonl"
OUT_DIR = SCRIPT_DIR / "processed"
DEFAULT_MERGED = OUT_DIR / "merged.jsonl"

_TAG_RE_UNDERSCORE = re.compile(r"^miner_requests_(\d+)\.jsonl$", re.IGNORECASE)
_TAG_RE_HYPHEN = re.compile(r"^miner_requests-(\d+)\.jsonl$", re.IGNORECASE)

def _stable_json_hash(obj: object) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_interval(spec: str) -> float:
    s = str(spec).strip().lower()
    if not s:
        raise ValueError("duration is empty")
    if s.endswith("h"):
        v = float(s[:-1].strip()) * 3600.0
    elif s.endswith("m"):
        v = float(s[:-1].strip()) * 60.0
    elif s.endswith("s"):
        v = float(s[:-1].strip())
    else:
        v = float(s)
    if v < 0:
        raise ValueError("duration must be non-negative")
    return v


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit, div in (("KiB", 1024), ("MiB", 1024**2), ("GiB", 1024**3)):
        x = n / div
        if x < 1024 or unit == "GiB":
            return f"{x:.2f} {unit}"
    return f"{n} B"


def _ensure_writable_dir(d: Path) -> bool:
    """Create directory if needed; require write access (for merge output parent, etc.)."""
    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"error: cannot create {d}: {e}", file=sys.stderr)
        return False
    if not os.access(d, os.W_OK):
        print(
            f"error: not writable: {d}\n"
            f"Fix ownership (e.g. created by root): sudo chown -R \"$USER:$USER\" {d}",
            file=sys.stderr,
        )
        return False
    return True


def _tag_from_miner_requests_name(name: str) -> int | None:
    m = _TAG_RE_UNDERSCORE.match(name)
    if m:
        return int(m.group(1))
    m = _TAG_RE_HYPHEN.match(name)
    if m:
        return int(m.group(1))
    return None


def _delete_empty_matching_logs(directory: Path, pattern: str) -> list[str]:
    """Unlink files in ``directory`` matching ``pattern`` with size 0. Returns basenames removed."""
    removed: list[str] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if not fnmatch.fnmatch(p.name, pattern):
            continue
        try:
            if p.stat().st_size == 0:
                p.unlink()
                removed.append(p.name)
        except OSError:
            continue
    return removed


def _delete_empty_matching_logs(directory: Path, glob_pattern: str) -> list[str]:
    """Unlink regular files with size 0 matching ``glob_pattern``. Returns basenames removed."""
    removed: list[str] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if not fnmatch.fnmatch(p.name, glob_pattern):
            continue
        try:
            if p.stat().st_size != 0:
                continue
            p.unlink()
            removed.append(p.name)
        except OSError:
            continue
    removed.sort()
    return removed


def _list_tagged_miner_files(directory: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        tag = _tag_from_miner_requests_name(p.name)
        if tag is None:
            continue
        found.append((tag, p))
    found.sort(key=lambda x: (x[0], x[1].name))
    return found


def _load_merged_hand_hashes(merged_path: Path) -> set[str]:
    """Fingerprints of hands already in merged.jsonl (streaming).

    Supports:
    - **Current format:** one JSON object per line = the hand dict itself.
    - **Legacy:** ``{"hand_hash": "...", "hand": {...}}`` (uses stored hash when present).
    """
    seen: set[str] = set()
    if not merged_path.is_file():
        return seen
    with merged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            hand_inner = obj.get("hand")
            if isinstance(hand_inner, dict):
                hh = obj.get("hand_hash")
                if isinstance(hh, str) and hh:
                    seen.add(hh)
                else:
                    seen.add(_stable_json_hash(hand_inner))
                continue
            seen.add(_stable_json_hash(obj))
    return seen


def merge_smallest_tagged_file(
    *,
    directory: Path,
    merged_path: Path,
    progress_every_lines: int,
) -> tuple[str, str, dict]:
    """
    If exactly one tagged log file exists, skip (miner's active file).

    If two or more exist, treat **largest tag** as the active log; pick the **smallest tag**
    among the others; append unique hands to ``merged_path``; delete that source file.

    Returns ``("ok"|"skip"|"err", message, stats)``.
    """
    stats = {
        "picked": "",
        "tag": -1,
        "source_lines": 0,
        "hands_in": 0,
        "appended": 0,
        "dup_skipped": 0,
        "skipped_lines": 0,
        "skipped_no_chunks": 0,
    }
    tagged = _list_tagged_miner_files(directory)
    if len(tagged) <= 1:
        return (
            "skip",
            "only one tagged miner_requests_* file (assumed live miner log; not touching)",
            stats,
        )

    max_tag = max(t for t, _ in tagged)
    eligible = [(t, p) for t, p in tagged if t < max_tag]
    if not eligible:
        return "skip", "no tagged file below current max tag", stats

    tag, src = min(eligible, key=lambda x: x[0])
    stats["picked"] = src.name
    stats["tag"] = tag

    try:
        sz = src.stat().st_size
    except OSError as e:
        return "err", f"stat {src}: {e}", stats

    seen = _load_merged_hand_hashes(merged_path)
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    file_size = max(sz, 1)
    bytes_read = 0
    batch_flush = 256

    try:
        with src.open("r", encoding="utf-8") as inp, merged_path.open(
            "a", encoding="utf-8"
        ) as out:
            for line in inp:
                raw = line
                line = line.strip()
                if not line:
                    continue
                bytes_read += len(raw.encode("utf-8"))
                stats["source_lines"] += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped_lines"] += 1
                    continue

                chunks = obj.get("chunks")
                if not isinstance(chunks, list) or not chunks:
                    stats["skipped_no_chunks"] += 1
                    continue

                for chunk in chunks:
                    if not isinstance(chunk, list):
                        continue
                    for hand in chunk:
                        if not isinstance(hand, dict):
                            continue
                        stats["hands_in"] += 1
                        hh = _stable_json_hash(hand)
                        if hh in seen:
                            stats["dup_skipped"] += 1
                            continue
                        seen.add(hh)
                        out.write(json.dumps(hand, ensure_ascii=False) + "\n")
                        stats["appended"] += 1
                        if stats["appended"] % batch_flush == 0:
                            out.flush()

                if (
                    progress_every_lines > 0
                    and stats["source_lines"] % progress_every_lines == 0
                ):
                    pct = min(100.0, 100.0 * bytes_read / file_size)
                    print(
                        f"    … merge {src.name}  lines={stats['source_lines']}  "
                        f"appended={stats['appended']}  ~{pct:.1f}% read",
                        flush=True,
                    )

            out.flush()
    except OSError as e:
        return "err", f"merge I/O error: {e}", stats

    try:
        src.unlink()
    except OSError as e:
        return (
            "err",
            f"merged ok but could not delete source {src}: {e}",
            stats,
        )

    return (
        "ok",
        f"merged {stats['appended']} new hands into {merged_path.name}; deleted {src.name}",
        stats,
    )


def _list_inplace_targets(directory: Path, pattern: str) -> list[Path]:
    """
    Tagged ``miner_requests_<n>`` / ``miner_requests-<n>`` files matching ``pattern``.

    If only one such file exists, returns [] (live miner log). Otherwise returns all paths
    whose tag is **strictly less than** the maximum tag (not the active file).
    """
    tagged: list[tuple[int, Path]] = []
    for tag, path in _list_tagged_miner_files(directory):
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        tagged.append((tag, path))

    if len(tagged) <= 1:
        return []

    max_tag = max(t for t, _ in tagged)
    out = [p for t, p in sorted(tagged, key=lambda x: (x[0], x[1].name)) if t < max_tag]
    return out


def dedupe_miner_request_jsonl_inplace(
    raw_path: Path,
    *,
    by_hands: bool,
    progress_every_lines: int,
    delete_source_if_no_output: bool,
) -> tuple[bool, str, dict]:
    """
    Read JSONL, write unique chunk/hand lines, atomically replace ``raw_path``.
    Output format: one JSON object per line (chunk_hash+hands or hand_hash+hand).
    """
    stats = {
        "jsonl_lines": 0,
        "skipped_lines": 0,
        "skipped_no_chunks": 0,
        "skipped_mismatch": 0,
        "skipped_bad_chunk": 0,
        "input_chunks": 0,
        "input_hands": 0,
        "unique_out": 0,
        "dup_dropped": 0,
    }
    seen: set[str] = set()
    tmp = raw_path.with_suffix(raw_path.suffix + ".dedupe.tmp")

    try:
        file_size = max(raw_path.stat().st_size, 1)
    except OSError as e:
        return False, f"cannot stat {raw_path}: {e}", stats

    bytes_read = 0
    try:
        with raw_path.open("r", encoding="utf-8") as inp, tmp.open("w", encoding="utf-8") as out:
            for line in inp:
                raw_line = line
                line = line.strip()
                if not line:
                    continue
                bytes_read += len(raw_line.encode("utf-8"))
                stats["jsonl_lines"] += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped_lines"] += 1
                    continue

                hashes = obj.get("chunk_hashes")
                chunks = obj.get("chunks")
                if not isinstance(chunks, list) or not chunks:
                    stats["skipped_no_chunks"] += 1
                    continue

                if not by_hands:
                    if not isinstance(hashes, list) or len(hashes) != len(chunks):
                        stats["skipped_mismatch"] += 1
                        continue
                    for chash, chunk in zip(hashes, chunks):
                        if not isinstance(chunk, list):
                            stats["skipped_bad_chunk"] += 1
                            continue
                        stats["input_chunks"] += 1
                        key = str(chash)
                        if key in seen:
                            stats["dup_dropped"] += 1
                            continue
                        seen.add(key)
                        rec = {"chunk_hash": key, "hands": chunk}
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        stats["unique_out"] += 1
                else:
                    for chunk in chunks:
                        if not isinstance(chunk, list):
                            stats["skipped_bad_chunk"] += 1
                            continue
                        for hand in chunk:
                            if not isinstance(hand, dict):
                                continue
                            stats["input_hands"] += 1
                            hh = _stable_json_hash(hand)
                            if hh in seen:
                                stats["dup_dropped"] += 1
                                continue
                            seen.add(hh)
                            rec = {"hand_hash": hh, "hand": hand}
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            stats["unique_out"] += 1

                if (
                    progress_every_lines > 0
                    and stats["jsonl_lines"] % progress_every_lines == 0
                ):
                    pct = min(100.0, 100.0 * bytes_read / file_size)
                    print(
                        f"    … {raw_path.name}  input_lines={stats['jsonl_lines']}  "
                        f"unique_written={stats['unique_out']}  ~read {pct:.1f}% of file size",
                        flush=True,
                    )

        if stats["unique_out"] == 0:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if stats["jsonl_lines"] > 0:
                if delete_source_if_no_output:
                    try:
                        raw_path.unlink()
                    except OSError as e:
                        return False, f"no records written; could not delete {raw_path}: {e}", stats
                    return (
                        True,
                        f"deleted {raw_path.name} (--in-place-delete-if-no-output; "
                        f"slim logs need --unique-by hand or --merge)",
                        stats,
                    )
                return (
                    False,
                    "no records written (all lines skipped). "
                    f"skipped_no_chunks={stats['skipped_no_chunks']} "
                    f"skipped_mismatch={stats['skipped_mismatch']} "
                    f"skipped_lines={stats['skipped_lines']}. "
                    "Slim logs without chunk_hashes need --unique-by hand or --merge. "
                    "Original file left unchanged.",
                    stats,
                )
            return True, f"{raw_path.name} (no non-empty input lines; unchanged)", stats

        tmp.replace(raw_path)
    except OSError as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False, f"I/O error: {e}", stats

    return True, str(raw_path), stats


def _run_inplace_cycle(
    *,
    directory: Path,
    pattern: str,
    by_hands: bool,
    progress_every_lines: int,
    delete_source_if_no_output: bool,
) -> tuple[int, int]:
    """Returns (files_ok, files_failed)."""
    targets = _list_inplace_targets(directory, pattern)
    if not targets:
        print(
            f"  (skip: single active log or no files matching {pattern!r} with numeric tag)",
            flush=True,
        )
        return 0, 0

    ok_c = 0
    fail_c = 0
    for path in targets:
        sz = path.stat().st_size
        print(
            f"  → in-place dedupe: {path.name} ({_fmt_bytes(sz)})",
            flush=True,
        )
        t0 = time.perf_counter()
        ok, msg, st = dedupe_miner_request_jsonl_inplace(
            path,
            by_hands=by_hands,
            progress_every_lines=progress_every_lines,
            delete_source_if_no_output=delete_source_if_no_output,
        )
        elapsed = time.perf_counter() - t0
        if ok:
            ok_c += 1
            mode = "hands" if by_hands else "chunks"
            print(
                f"  ✓ {path.name}  unique={st['unique_out']} ({mode})  "
                f"in_lines={st['jsonl_lines']} dup={st['dup_dropped']}  "
                f"{elapsed:.2f}s",
                flush=True,
            )
            if st["unique_out"] == 0:
                print(f"     {msg}", flush=True)
        else:
            fail_c += 1
            print(f"  ✗ {path.name}  FAILED: {msg}", flush=True)
    return ok_c, fail_c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merge",
        action="store_true",
        help=(
            "among rotated logs (tags < current max): smallest tag → append unique hands to "
            "merged.jsonl → delete that file; skip if only one tagged file"
        ),
    )
    p.add_argument(
        "--merged-output",
        type=Path,
        default=DEFAULT_MERGED,
        help=f"path for cumulative merged hands (default: {DEFAULT_MERGED})",
    )
    p.add_argument(
        "--directory",
        type=Path,
        default=SCRIPT_DIR,
        help="directory for in-place glob (default: this script's folder)",
    )
    p.add_argument(
        "--glob",
        dest="glob_pattern",
        default=DEFAULT_GLOB,
        help=f"filename glob for rotated logs (default: {DEFAULT_GLOB})",
    )
    p.add_argument(
        "--initial-wait",
        type=str,
        default="2h",
        metavar="DURATION",
        help="sleep before first cycle, merge or in-place (default: 2h)",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1h",
        metavar="DURATION",
        help="repeat interval between cycles (default: 1h)",
    )
    p.add_argument(
        "--progress-every-lines",
        type=int,
        default=200,
        metavar="N",
        help="print progress every N input JSONL lines (0=disable)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="single cycle then exit (skip long first wait with --initial-wait 0)",
    )
    p.add_argument(
        "--unique-by",
        choices=("chunk", "hand"),
        default="chunk",
        help="dedupe by chunk_hashes+chunks or per-hand hash",
    )
    p.add_argument(
        "--in-place-delete-if-no-output",
        action="store_true",
        help=(
            "in-place only: if every input line is skipped (e.g. slim logs without "
            "chunk_hashes), delete the source file instead of leaving it unchanged"
        ),
    )
    p.add_argument(
        "--delete-empty-logs",
        action="store_true",
        help=(
            "after each cycle, delete zero-byte files in --directory matching --glob "
            "(e.g. leftover empty miner_requests_*.jsonl)"
        ),
    )
    args = p.parse_args()

    try:
        initial_wait_sec = parse_interval(args.initial_wait)
        interval_sec = parse_interval(args.interval)
    except ValueError as e:
        print(f"error: invalid duration: {e}", file=sys.stderr)
        return 1

    by_hands = args.unique_by == "hand"

    # ----- Merge mode (append unique hands → merged.jsonl, delete source) -----
    if args.merge:
        directory = args.directory.expanduser().resolve()
        merged_path = args.merged_output.expanduser().resolve()
        if not directory.is_dir():
            print(f"error: not a directory: {directory}", file=sys.stderr)
            return 1
        if not _ensure_writable_dir(merged_path.parent):
            return 1

        print(
            f"Merge mode (hand-level dedupe)\n"
            f"Scan directory: {directory}\n"
            f"Merged output: {merged_path}\n"
            f"Pick: smallest tag among files with tag < max tag "
            f"(miner_requests_<n>.jsonl / miner_requests-<n>.jsonl); "
            f"skip if only one tagged file\n"
            f"Initial wait: {args.initial_wait!r} ({initial_wait_sec:.0f}s)\n"
            f"Repeat interval: {args.interval!r} ({interval_sec:.0f}s)\n"
            f"(Ctrl+C to stop)\n",
            flush=True,
        )

        cycle = 0
        try:
            while True:
                cycle += 1
                if cycle == 1 and not args.once and initial_wait_sec > 0:
                    w0 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    print(
                        f"[cycle {cycle}] {w0}  waiting {initial_wait_sec:.0f}s before first merge…",
                        flush=True,
                    )
                    time.sleep(initial_wait_sec)

                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                print(f"\n[cycle {cycle}] {ts}  merge pass…", flush=True)
                t0 = time.perf_counter()
                status, msg, st = merge_smallest_tagged_file(
                    directory=directory,
                    merged_path=merged_path,
                    progress_every_lines=max(0, int(args.progress_every_lines)),
                )
                elapsed = time.perf_counter() - t0

                if status == "ok":
                    print(
                        f"  ✓ tag={st['tag']} file={st['picked']}  "
                        f"hands_in={st['hands_in']} appended={st['appended']} "
                        f"dup_skip={st['dup_skipped']}  {elapsed:.2f}s\n"
                        f"  {msg}",
                        flush=True,
                    )
                    rc = 0
                elif status == "skip":
                    print(f"  — skip: {msg}  ({elapsed:.2f}s)", flush=True)
                    rc = 0
                else:
                    print(f"  ✗ FAIL: {msg}  ({elapsed:.2f}s)", flush=True)
                    rc = 1

                print(
                    f"[cycle {cycle}] done in {elapsed:.2f}s",
                    flush=True,
                )

                if args.delete_empty_logs:
                    gone = _delete_empty_matching_logs(directory, args.glob_pattern)
                    if gone:
                        print(f"  deleted empty log(s): {', '.join(gone)}", flush=True)

                if args.once:
                    return rc

                print(f"Sleeping {interval_sec:.0f}s…\n", flush=True)
                time.sleep(max(0.0, interval_sec))
        except KeyboardInterrupt:
            print("\nStopped by user.", flush=True)
            return 0

    # ----- In-place mode (default) -----
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    print(
        f"In-place dedupe mode\n"
        f"Directory: {directory}\n"
        f"Glob: {args.glob_pattern!r} (numeric-tagged miner_requests_* only)\n"
        f"Targets: tags strictly less than max tag; skip if only one matching tagged file\n"
        f"Initial wait: {args.initial_wait!r} ({initial_wait_sec:.0f}s)\n"
        f"Repeat interval: {args.interval!r} ({interval_sec:.0f}s)\n"
        f"Dedupe: {'hand' if by_hands else 'chunk (chunk_hashes)'}\n"
        f"(Ctrl+C to stop)\n",
        flush=True,
    )

    cycle = 0
    try:
        while True:
            cycle += 1
            if cycle == 1 and not args.once and initial_wait_sec > 0:
                w0 = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                print(
                    f"[cycle {cycle}] {w0}  waiting {initial_wait_sec:.0f}s before first run…",
                    flush=True,
                )
                time.sleep(initial_wait_sec)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n[cycle {cycle}] {ts}  starting in-place pass…", flush=True)
            t0 = time.perf_counter()
            ok_c, fail_c = _run_inplace_cycle(
                directory=directory,
                pattern=args.glob_pattern,
                by_hands=by_hands,
                progress_every_lines=max(0, int(args.progress_every_lines)),
                delete_source_if_no_output=bool(args.in_place_delete_if_no_output),
            )
            elapsed = time.perf_counter() - t0
            print(
                f"[cycle {cycle}] done in {elapsed:.2f}s  "
                f"files_ok={ok_c} files_failed={fail_c}",
                flush=True,
            )

            if args.delete_empty_logs:
                gone = _delete_empty_matching_logs(directory, args.glob_pattern)
                if gone:
                    print(f"  deleted empty log(s): {', '.join(gone)}", flush=True)

            if args.once:
                return 1 if fail_c else 0

            print(
                f"Sleeping {interval_sec:.0f}s until next cycle…\n",
                flush=True,
            )
            time.sleep(max(0.0, interval_sec))
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
