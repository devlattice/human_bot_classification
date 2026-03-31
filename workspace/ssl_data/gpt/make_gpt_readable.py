#!/usr/bin/env python3
"""
Build compact GPT user-message text files from score-split JSON (e.g. score_medium.json).

Each top-level array element should have a \"chunk\" list of hand dicts. The prompt
includes \"chunk_hash\" when present (for traceability / join); if absent,
\"chunk_hash: (none)\" is written. Output files are named prompt_NNNNNN.txt using
the source JSON array index.

CPU-oriented: one pass per record over actions (Counter); no full-chunk stringification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent

# Default pricing (GPT-4o mini class, USD / 1M tokens) — override with env or flags if you add them.
DEFAULT_USD_PER_1M_INPUT = 0.15
DEFAULT_USD_PER_1M_OUTPUT = 0.60
# Minimal JSON reply: {"label":0}
DEFAULT_ASSUMED_OUTPUT_TOKENS = 12


def _street_abbr(street: Any) -> str:
    s = str(street or "")[:2].lower()
    if s == "pr" or str(street).lower().startswith("preflop"):
        return "pr"
    if s == "fl" or str(street).lower().startswith("flop"):
        return "fl"
    if s == "tu" or str(street).lower().startswith("turn"):
        return "tu"
    if s == "ri" or str(street).lower().startswith("river"):
        return "ri"
    return str(street)[:2] if street else "?"


def _hand_shorthand(hand: dict[str, Any]) -> str:
    parts: list[str] = []
    for a in hand.get("actions") or []:
        if not isinstance(a, dict):
            continue
        seat = a.get("actor_seat", "?")
        ab = _street_abbr(a.get("street"))
        at = a.get("action_type") or "?"
        parts.append(f"s{seat}:{ab}:{at}")
    return " | ".join(parts)


def _aggregate_chunk(chunk: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], int]:
    act_c: Counter[str] = Counter()
    st_c: Counter[str] = Counter()
    n_act = 0
    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        for a in hand.get("actions") or []:
            if not isinstance(a, dict):
                continue
            t = a.get("action_type")
            if t is not None:
                act_c[str(t)] += 1
            s = a.get("street")
            if s is not None:
                st_c[str(s)] += 1
            n_act += 1
    return act_c, st_c, n_act


def _first_hand_meta(chunk: list[dict[str, Any]]) -> dict[str, Any]:
    for hand in chunk:
        if isinstance(hand, dict) and isinstance(hand.get("metadata"), dict):
            return hand["metadata"]
    return {}


def build_user_message(
    chunk: list[dict[str, Any]],
    *,
    chunk_hash: str | None,
    risk_score: float | None,
    max_shorthand_hands: int,
    include_outlier_placeholder: bool,
) -> str:
    meta = _first_hand_meta(chunk)
    game_type = meta.get("game_type", "Hold'em")
    limit_type = meta.get("limit_type", "No Limit")
    bb = meta.get("bb", "?")
    max_seats = meta.get("max_seats", "?")

    act_c, st_c, n_act = _aggregate_chunk(chunk)
    n_hands = len([h for h in chunk if isinstance(h, dict)])

    # Stable ordering for counts (readable + reproducible).
    act_items = sorted(act_c.items(), key=lambda x: (-x[1], x[0]))
    st_items = sorted(st_c.items(), key=lambda x: (-x[1], x[0]))
    act_str = ", ".join(f"{k}={v}" for k, v in act_items) if act_items else "(none)"
    st_str = ", ".join(f"{k}={v}" for k, v in st_items) if st_items else "(none)"

    pre = st_c.get("preflop", 0)
    note = (
        "most action volume is preflop; postflop depth is limited on average."
        if n_act > 0 and pre / n_act >= 0.75
        else "mixed preflop/postflop action."
    )

    h = chunk_hash.strip() if isinstance(chunk_hash, str) and chunk_hash.strip() else None
    header = [
        f"chunk_hash: {h}" if h is not None else "chunk_hash: (none)",
        "",
    ]

    lines: list[str] = [
        *header,
        "=== CHUNK_SUMMARY (derived from raw hands; not labels) ===",
        f"- Game: {game_type} {limit_type}, {max_seats}-max, bb={bb}",
        f"- Hands in chunk: {n_hands}",
        f"- Total actions (all hands): {n_act}",
        f"- Action mix (counts): {act_str}",
        f"- Street mix (counts): {st_str}",
        f"- Note: {note}",
        "",
    ]
    if risk_score is not None and risk_score == risk_score:  # not NaN
        lines.append(
            "Teacher score (informational only — do NOT treat as ground truth): "
            f"risk_score≈{float(risk_score):.4f}"
        )
        lines.append("")

    max_h = max(0, int(max_shorthand_hands))
    n_slots = len(chunk)
    written = 0
    for hi, hand in enumerate(chunk):
        if written >= max_h:
            break
        if not isinstance(hand, dict):
            continue
        sh = _hand_shorthand(hand)
        lines.append(
            f"=== ACTION SHORTHAND (chunk index {hi} of {n_slots}; "
            f"shorthand {written + 1}/{max_h}; seat:street:action) ==="
        )
        lines.append(sh)
        lines.append("")
        written += 1

    if include_outlier_placeholder:
        lines.extend(
            [
                "=== OPTIONAL OUTLIER HINTS (add z-scores vs your training pool offline) ===",
                "- (example) fold_ratio_mean z=+1.2 vs human reference",
                "- (omit if unknown)",
                "",
            ]
        )

    lines.extend(
        [
            "=== TASK ===",
            'Return JSON only: {"label":0} or {"label":1} per the system definitions.',
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (array_index, record). Tries ijson first; falls back to json.load on failure."""
    try:
        import ijson  # type: ignore

        try:
            with path.open("rb") as f:
                for idx, item in enumerate(ijson.items(f, "item")):
                    if isinstance(item, dict):
                        yield idx, item
            return
        except Exception as e:
            print(
                f"[make_gpt_readable] ijson failed ({type(e).__name__}: {e}); using json.load",
                file=sys.stderr,
            )
    except ImportError:
        pass

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            yield idx, item


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _token_backend_name() -> str:
    try:
        import tiktoken  # noqa: F401

        return "tiktoken(gpt-4o)"
    except Exception:
        return "heuristic(~chars/4)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build compact GPT-readable prompts from score JSON.")
    ap.add_argument(
        "--input-json",
        type=Path,
        default=Path("workspace/datasets/ssl_data/split_out/score_medium.json"),
        help="JSON array of {chunk: [...], risk_score?, chunk_hash?}",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for prompt_NNNNNN.txt files",
    )
    ap.add_argument(
        "--max-shorthand-hands",
        type=int,
        default=1,
        help="Number of hands to expand into ACTION SHORTHAND (default 1; increase for more context).",
    )
    ap.add_argument(
        "--include-outlier-placeholder",
        action="store_true",
        help="Include optional outlier-hint template (extra tokens).",
    )
    ap.add_argument(
        "--system-prompt-file",
        type=Path,
        default=SCRIPT_DIR / "system_prompt_stub.txt",
        help="Path to system prompt text (for token / cost estimate only).",
    )
    ap.add_argument(
        "--usd-per-1m-input",
        type=float,
        default=DEFAULT_USD_PER_1M_INPUT,
        help="USD per 1M input tokens for cost estimate (default: GPT-4o mini scale).",
    )
    ap.add_argument(
        "--usd-per-1m-output",
        type=float,
        default=DEFAULT_USD_PER_1M_OUTPUT,
        help="USD per 1M output tokens for cost estimate.",
    )
    ap.add_argument(
        "--assumed-output-tokens",
        type=int,
        default=DEFAULT_ASSUMED_OUTPUT_TOKENS,
        help="Assumed completion tokens per API call for cost estimate.",
    )
    args = ap.parse_args()

    in_path = args.input_json.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not in_path.is_file():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    system_text = ""
    sp = args.system_prompt_file.expanduser().resolve()
    if sp.is_file():
        system_text = sp.read_text(encoding="utf-8")

    n_written = 0
    total_user_tokens = 0
    skipped = 0

    for idx, rec in _iter_records(in_path):
        chunk = rec.get("chunk")
        if not isinstance(chunk, list) or not chunk:
            skipped += 1
            continue
        rs = rec.get("risk_score")
        rs_f: float | None
        try:
            rs_f = float(rs)
        except (TypeError, ValueError):
            rs_f = None

        raw_hash = rec.get("chunk_hash")
        ch_str = str(raw_hash).strip() if raw_hash is not None else None

        text = build_user_message(
            chunk,
            chunk_hash=ch_str,
            risk_score=rs_f,
            max_shorthand_hands=max(0, int(args.max_shorthand_hands)),
            include_outlier_placeholder=bool(args.include_outlier_placeholder),
        )
        out_file = prompts_dir / f"prompt_{idx:06d}.txt"
        out_file.write_text(text, encoding="utf-8")
        n_written += 1
        total_user_tokens += _count_tokens(text)

    system_tokens = _count_tokens(system_text) if system_text else 0
    per_call_user_avg = total_user_tokens / n_written if n_written else 0
    # Each API call: system (once) + user — for batch, system repeated per request in chat API:
    input_tokens_total = n_written * system_tokens + total_user_tokens
    avg_input_per_call = system_tokens + per_call_user_avg

    out_tok = max(0, int(args.assumed_output_tokens))
    output_tokens_total = n_written * out_tok

    cost_in = (input_tokens_total / 1_000_000.0) * float(args.usd_per_1m_input)
    cost_out = (output_tokens_total / 1_000_000.0) * float(args.usd_per_1m_output)
    cost_total = cost_in + cost_out

    tok_method = _token_backend_name()

    print(
        f"[make_gpt_readable] records_written={n_written} skipped_empty={skipped} "
        f"output={prompts_dir}",
        file=sys.stderr,
    )
    if n_written == 0:
        print(
            "[make_gpt_readable] Warning: no prompts written (empty chunk lists or no dict records?).",
            file=sys.stderr,
        )
    # Primary estimates on stdout (easy to copy / pipe).
    print(f"token_backend={tok_method}")
    print(f"system_prompt_tokens≈{system_tokens}")
    print(f"avg_user_message_tokens≈{per_call_user_avg:.2f}")
    print(f"avg_input_tokens_per_api_call≈{avg_input_per_call:.2f}  # system + user (repeated per call)")
    print(f"total_input_tokens≈{input_tokens_total}  # {n_written} calls")
    print(f"total_output_tokens≈{output_tokens_total}  # assumed {out_tok} tokens/call")
    print(
        f"estimated_cost_USD≈{cost_total:.6f}  "
        f"(input≈{cost_in:.6f} + output≈{cost_out:.6f} at "
        f"${args.usd_per_1m_input}/1M in, ${args.usd_per_1m_output}/1M out)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
