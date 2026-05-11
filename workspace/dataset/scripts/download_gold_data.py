#!/usr/bin/env python3
"""Download Poker44 released gold benchmark chunks into local JSON files.

Example:
    python workspace/dataset/scripts/download_gold_data.py \
      --out-dir workspace/dataset/source/gold_dataset \
      --max-days 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"


def fetch_json(url: str, timeout: int = 60) -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "p44-gold-builder/1.0"})
    with urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = response.read().decode(charset)
    return json.loads(payload)


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "p44-gold-builder/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def extract_release_items(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("releases", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        for key in ("items", "releases", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if "sourceDate" in payload:
            return [payload]
    raise ValueError("Could not parse releases payload shape.")


def unique_source_dates(items: Iterable[dict]) -> List[str]:
    dates = sorted({str(item.get("sourceDate", "")).strip() for item in items if item.get("sourceDate")})
    return [d for d in dates if d]


def write_json(path: Path, obj: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(obj, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local gold dataset from Poker44 benchmark API releases.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where downloaded daily JSON files are stored.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Benchmark API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="Sleep between day downloads to avoid hammering API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when the day file already exists.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="If > 0, download only the latest N days.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    print(f"[start] out_dir={out_dir}", flush=True)

    releases_url = f"{args.base_url.rstrip('/')}/releases"
    try:
        releases_payload = fetch_json(releases_url, timeout=args.timeout_seconds)
        release_items = extract_release_items(releases_payload)
        source_dates = unique_source_dates(release_items)
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] failed to fetch releases: {exc}", file=sys.stderr)
        return 1

    if not source_dates:
        print("[warn] no source dates found in releases response.")
        return 0

    if args.max_days > 0:
        source_dates = source_dates[-args.max_days :]

    write_json(out_dir / "releases.json", releases_payload)
    print(
        f"[releases] total_available={len(unique_source_dates(release_items))} selected={len(source_dates)}",
        flush=True,
    )

    downloaded = 0
    skipped = 0
    failed = 0
    bytes_downloaded = 0

    for idx, source_date in enumerate(source_dates, start=1):
        t0 = time.time()
        day_path = out_dir / f"{source_date}.json"
        if day_path.exists() and not args.force:
            if day_path.stat().st_size > 0:
                skipped += 1
                print(f"[{idx}/{len(source_dates)}] skip {source_date} (already exists)", flush=True)
                continue
            print(f"[{idx}/{len(source_dates)}] re-download {source_date} (empty file found)", flush=True)

        day_url = f"{args.base_url.rstrip('/')}/chunks?sourceDate={source_date}"
        try:
            print(f"[{idx}/{len(source_dates)}] downloading {source_date} ...", flush=True)
            day_bytes = fetch_bytes(day_url, timeout=args.timeout_seconds)
            if not day_bytes:
                raise ValueError("received empty response body")
            tmp_path = day_path.with_suffix(day_path.suffix + ".tmp")
            tmp_path.write_bytes(day_bytes)
            tmp_path.replace(day_path)
            downloaded += 1
            bytes_downloaded += len(day_bytes)
            elapsed = max(1e-6, time.time() - t0)
            size_mb = len(day_bytes) / (1024 * 1024)
            speed_mb_s = size_mb / elapsed
            print(
                f"[{idx}/{len(source_dates)}] saved {source_date} -> {day_path.name} "
                f"({size_mb:.1f} MB in {elapsed:.1f}s, {speed_mb_s:.2f} MB/s)",
                flush=True,
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            failed += 1
            print(f"[{idx}/{len(source_dates)}] fail {source_date}: {exc}", file=sys.stderr, flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    total_elapsed = max(1e-6, time.time() - run_started)
    summary = {
        "base_url": args.base_url,
        "total_dates": len(source_dates),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "bytes_downloaded": bytes_downloaded,
        "elapsed_seconds": round(total_elapsed, 3),
        "generated_at_unix": int(time.time()),
    }
    write_json(out_dir / "_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
