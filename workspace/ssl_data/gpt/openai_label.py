#!/usr/bin/env python3
"""
Call OpenAI chat completions for each prompt_*.txt, merge labels onto --source records, write refine JSON.

Also writes the full --source JSON array back with an extra int field ``gpt_score`` (0 or 1) on each
successfully labeled row (same index as prompt_NNNNNN.txt). Use --in-place or --source-out.

This file is named ``openai_label.py`` (not ``openai.py``) so it does not shadow the PyPI ``openai``
package when that directory is on ``sys.path``.

CPU / rate-limit friendly: one request at a time, optional sleep between calls.
Requires: pip install openai
Env: OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

PROMPT_NAME_RE = re.compile(r"^prompt_(\d+)\.txt$", re.IGNORECASE)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _prompt_paths(input_data: Path) -> list[Path]:
    prompts_dir = input_data / "prompts"
    if prompts_dir.is_dir():
        base = prompts_dir
    else:
        base = input_data
    paths = sorted(base.glob("prompt_*.txt"))
    return paths


def _index_from_prompt_path(p: Path) -> int | None:
    m = PROMPT_NAME_RE.match(p.name)
    return int(m.group(1)) if m else None


def _load_source_subset(source_path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    if not indices:
        return {}
    max_i = max(indices)

    try:
        import ijson  # type: ignore

        out: dict[int, dict[str, Any]] = {}
        with source_path.open("rb") as f:
            for idx, item in enumerate(ijson.items(f, "item")):
                if not isinstance(item, dict):
                    continue
                if idx in indices:
                    out[idx] = item
                if idx >= max_i and len(out) == len(indices):
                    break
        return out
    except ImportError:
        _log("[gpt-label] ijson not installed; loading full --source with json.load (high RAM if file is large).")
        with source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{source_path} must be a JSON array")
        return {i: data[i] for i in indices if isinstance(i, int) and 0 <= i < len(data) and isinstance(data[i], dict)}


def _load_full_source_array(source_path: Path) -> list[Any]:
    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{source_path} must be a JSON array")
    return data


def _write_source_with_gpt_scores(
    source_path: Path,
    dest_path: Path,
    index_to_score: dict[int, int],
) -> None:
    data = _load_full_source_array(source_path)
    applied = 0
    for idx, score in index_to_score.items():
        if not isinstance(idx, int) or idx < 0 or idx >= len(data):
            continue
        rec = data[idx]
        if not isinstance(rec, dict):
            continue
        rec["gpt_score"] = int(score)
        applied += 1
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")
    _log(f"[gpt-label] wrote source with gpt_score on {applied} row(s) -> {dest_path}")


def _read_system_prompt(path: Path | None) -> str:
    p = path if path is not None else SCRIPT_DIR / "system_prompt_stub.txt"
    p = p.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"system prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _parse_label_json(content: str) -> int | None:
    t = (content or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", t)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    y = obj.get("label")
    if y in (0, 1):
        return int(y)
    if y in ("0", "1"):
        return int(y)
    return None


def _call_openai(
    *,
    user_text: str,
    system_text: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> tuple[int | None, str]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Missing package 'openai'. Install with: pip install openai\n" f"({e})"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key, timeout=timeout)
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            label = _parse_label_json(raw)
            if label is not None:
                return label, raw
            last_err = f"unparseable content: {raw[:200]!r}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            wait = min(30.0, 1.5**attempt)
            _log(f"[gpt-label] retry {attempt}/{max_retries} after error ({last_err}); sleeping {wait:.1f}s")
            time.sleep(wait)
    return None, last_err


def _build_refine_record(source_rec: dict[str, Any], label: int) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "chunk": source_rec["chunk"]}
    if "chunk_hash" in source_rec:
        out["chunk_hash"] = source_rec["chunk_hash"]
    if "risk_score" in source_rec:
        out["risk_score"] = source_rec["risk_score"]
    return out


def main() -> int:
    repo_root = SCRIPT_DIR.parent.parent.parent.parent
    default_input = repo_root / "workspace/datasets/ssl_data/gpt/prompts_out"
    default_source = repo_root / "workspace/datasets/ssl_data/split_out/score_medium.json"
    default_output = default_input / "refined"

    ap = argparse.ArgumentParser(description="Label prompt files via OpenAI; write refine-score-json.json for build_dataset.")
    ap.add_argument(
        "--input-data",
        "--input-dta",
        dest="input_data",
        type=Path,
        default=default_input,
        help="Directory containing prompts/ (or prompt_*.txt directly).",
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="JSON array (e.g. score_medium.json) aligned by prompt index.",
    )
    ap.add_argument(
        "--source-out",
        type=Path,
        default=None,
        help=(
            "Write full source JSON with gpt_score added on labeled rows. "
            "Default: <source_stem>_gptscore.json next to --source. Ignored if --in-place."
        ),
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Write augmented source JSON over --source (use with care).",
    )
    ap.add_argument(
        "--output",
        "--out-put",
        "--out_put",
        dest="output",
        type=Path,
        default=default_output,
        help="Output directory (writes refine-score-json.json).",
    )
    ap.add_argument(
        "--system-prompt-file",
        type=Path,
        default=SCRIPT_DIR / "system_prompt_stub.txt",
        help="System message for the model.",
    )
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat completion model name.")
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Pause after each successful request (rate limiting / CPU idle).",
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout (seconds).")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per prompt on failure.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N prompts (0 = all).")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call API; print planned work and write no file.",
    )
    args = ap.parse_args()

    input_data = args.input_data.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_data.is_dir():
        _log(f"[gpt-label] error: --input-data is not a directory: {input_data}")
        return 1
    if not source_path.is_file():
        _log(f"[gpt-label] error: --source not found: {source_path}")
        return 1

    paths = _prompt_paths(input_data)
    indexed: list[tuple[int, Path]] = []
    for p in paths:
        idx = _index_from_prompt_path(p)
        if idx is None:
            _log(f"[gpt-label] skip (bad name): {p.name}")
            continue
        indexed.append((idx, p))
    indexed.sort(key=lambda x: x[0])

    if args.limit and args.limit > 0:
        indexed = indexed[: args.limit]

    if not indexed:
        _log("[gpt-label] no prompt_*.txt files found.")
        return 1

    indices = {i for i, _ in indexed}
    _log(f"[gpt-label] prompts={len(indexed)} unique_indices={len(indices)} input_data={input_data}")

    _log("[gpt-label] loading source rows for those indices …")
    t0 = time.perf_counter()
    source_by_idx = _load_source_subset(source_path, indices)
    _log(f"[gpt-label] source load done in {time.perf_counter() - t0:.2f}s (records={len(source_by_idx)})")

    missing = sorted(indices - set(source_by_idx.keys()))
    if missing:
        _log(f"[gpt-label] warning: {len(missing)} source index(es) missing (first 20): {missing[:20]}")

    system_text = _read_system_prompt(args.system_prompt_file)
    _log(f"[gpt-label] model={args.model!r} system_prompt_chars={len(system_text)}")

    if args.in_place:
        source_out = source_path
    elif args.source_out is not None:
        source_out = args.source_out.expanduser().resolve()
    else:
        source_out = source_path.with_name(f"{source_path.stem}_gptscore{source_path.suffix}")

    if args.dry_run:
        for i, (pidx, p) in enumerate(indexed, start=1):
            ok = "ok" if pidx in source_by_idx else "NO_SOURCE"
            _log(f"[gpt-label] dry-run {i}/{len(indexed)} index={pidx} {ok} {p.name}")
        _log(f"[gpt-label] dry-run finished; no API calls. Would write source -> {source_out}")
        _log("[gpt-label] dry-run: refine-score-json.json not written.")
        return 0

    refined: list[dict[str, Any]] = []
    index_to_gpt_score: dict[int, int] = {}
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for n, (pidx, ppath) in enumerate(indexed, start=1):
        rec = source_by_idx.get(pidx)
        if rec is None:
            _log(f"[gpt-label] {n}/{len(indexed)} index={pidx} SKIP (missing source row) {ppath.name}")
            n_skip += 1
            continue
        chunk = rec.get("chunk")
        if not isinstance(chunk, list) or not chunk:
            _log(f"[gpt-label] {n}/{len(indexed)} index={pidx} SKIP (empty chunk) {ppath.name}")
            n_skip += 1
            continue

        user_text = ppath.read_text(encoding="utf-8")
        _log(f"[gpt-label] {n}/{len(indexed)} index={pidx} calling API … ({ppath.name}, user_chars={len(user_text)})")
        t_req = time.perf_counter()
        label, info = _call_openai(
            user_text=user_text,
            system_text=system_text,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        dt = time.perf_counter() - t_req
        if label is None:
            _log(f"[gpt-label] {n}/{len(indexed)} index={pidx} FAIL ({info}) after {dt:.2f}s")
            n_fail += 1
            continue

        refined.append(_build_refine_record(rec, label))
        index_to_gpt_score[pidx] = label
        n_ok += 1
        _log(f"[gpt-label] {n}/{len(indexed)} index={pidx} label={label} ok ({dt:.2f}s)")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "refine-score-json.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False)
        f.write("\n")

    if index_to_gpt_score:
        _write_source_with_gpt_scores(source_path, source_out, index_to_gpt_score)
    else:
        _log("[gpt-label] no successful labels; skipping source JSON write (gpt_score).")

    _log(
        f"[gpt-label] done wrote {out_path} records={len(refined)} "
        f"ok={n_ok} skipped={n_skip} failed={n_fail}"
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
