#!/usr/bin/env python3
"""Convert bot PHHS dumps into Poker44 canonical schema (multi-seat configurable).

Example:
python workspace/dataset/source/scripts/bot.py \
  --sample-json workspace/dataset/source/data/poker_hands_train.json \
  --input-dir workspace/dataset/source/bot \
  --out-dir workspace/dataset/source/data/bot \
  --max-hands-per-folder 50000 \
  --min-players 2 \
  --max-players 6 \
  --min-actions 4 \
  --seed 42 \
  --progress-every-blocks 5000
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None


_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_BLOCK_RE = re.compile(r"^\[\d+\]\s*$")
_PHHS_GLOB = "*.phhs"


@dataclass
class Args:
    sample_json: Path
    input_dir: Path
    out_dir: Path
    max_hands_per_folder: int
    min_players: int
    max_players: int
    min_actions: int
    seed: int
    progress_every_blocks: int
    parquet_batch_size: int
    write_parquet: bool


def _stable_uid(raw_player: str) -> str:
    digest = hashlib.sha256(raw_player.encode("utf-8")).hexdigest()
    return f"p_{digest}"


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_rhs(raw: str) -> Any:
    value = raw.strip()
    if _TIME_RE.match(value):
        return value
    value = value.replace("true", "True").replace("false", "False").replace("null", "None")
    return ast.literal_eval(value)


def _iter_phhs_blocks(path: Path) -> Iterator[Dict[str, Any]]:
    current: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if _BLOCK_RE.match(line):
                if current:
                    yield current
                    current = {}
                continue
            if "=" not in line:
                continue
            key, rhs = line.split("=", 1)
            try:
                current[key.strip()] = _parse_rhs(rhs.strip())
            except Exception:
                continue
    if current:
        yield current


def _board_streets(actions: Sequence[str]) -> List[Dict[str, Any]]:
    streets: List[Dict[str, Any]] = []
    labels = ["flop", "turn", "river"]
    idx = 0
    board_cards: List[str] = []
    for token in actions:
        parts = token.split()
        if len(parts) < 3 or parts[0] != "d" or parts[1] != "db":
            continue
        cards = parts[2]
        if len(cards) % 2 != 0:
            continue
        split_cards = [cards[i : i + 2] for i in range(0, len(cards), 2)]
        if idx == 0 and len(split_cards) >= 3:
            board_cards = split_cards[:3]
            streets.append({"street": labels[idx], "board_cards": board_cards.copy()})
            idx += 1
            continue
        if not split_cards:
            continue
        board_cards = board_cards + [split_cards[0]]
        if idx < len(labels):
            streets.append({"street": labels[idx], "board_cards": board_cards.copy()})
            idx += 1
    return streets


def _canonicalize_hand(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    players_raw = block.get("players") or []
    seats_raw = block.get("seats") or []
    stacks_raw = block.get("starting_stacks") or []
    winnings_raw = block.get("winnings") or []
    actions_raw = block.get("actions") or []
    blinds = block.get("blinds_or_straddles") or []

    if not isinstance(players_raw, list) or not isinstance(actions_raw, list):
        return None
    if len(players_raw) < 2:
        return None

    sb = _to_float(blinds[0]) if len(blinds) > 0 else 0.0
    bb = _to_float(blinds[1]) if len(blinds) > 1 else 0.0
    if bb <= 0:
        return None

    players: List[Dict[str, Any]] = []
    uid_by_pos: Dict[int, str] = {}
    seat_by_pos: Dict[int, int] = {}
    contrib_by_seat: Dict[int, float] = {}

    for i, raw_player in enumerate(players_raw, start=1):
        seat = _to_int(seats_raw[i - 1], i) if i - 1 < len(seats_raw) else i
        starting_stack = _to_float(stacks_raw[i - 1], 0.0) if i - 1 < len(stacks_raw) else 0.0
        uid = _stable_uid(str(raw_player))
        players.append(
            {
                "player_uid": uid,
                "seat": seat,
                "starting_stack": round(starting_stack, 4),
                "hole_cards": None,
                "showed_hand": False,
            }
        )
        uid_by_pos[i] = uid
        seat_by_pos[i] = seat
        contrib_by_seat[seat] = 0.0

    actions: List[Dict[str, Any]] = []
    pot = 0.0
    action_id = 1
    current_street = "preflop"
    current_bet = 0.0
    showdown_seen = False

    def append_action(
        actor_seat: int,
        action_type: str,
        amount: float,
        raise_to: Optional[float],
        call_to: Optional[float],
    ) -> None:
        nonlocal pot, action_id
        amount = max(0.0, round(amount, 4))
        pot_before = round(pot, 4)
        pot_after = round(pot_before + amount, 4)
        actions.append(
            {
                "action_id": str(action_id),
                "street": current_street,
                "actor_seat": actor_seat,
                "action_type": action_type,
                "amount": amount,
                "raise_to": None if raise_to is None else round(raise_to, 4),
                "call_to": None if call_to is None else round(call_to, 4),
                "normalized_amount_bb": round(amount / bb, 4),
                "pot_before": pot_before,
                "pot_after": pot_after,
            }
        )
        action_id += 1
        pot = pot_after

    if sb > 0:
        seat = seat_by_pos.get(1, 1)
        append_action(seat, "small_blind", sb, None, None)
        contrib_by_seat[seat] = round(contrib_by_seat.get(seat, 0.0) + sb, 4)
    if bb > 0:
        seat = seat_by_pos.get(2, 2)
        append_action(seat, "big_blind", bb, None, None)
        contrib_by_seat[seat] = round(contrib_by_seat.get(seat, 0.0) + bb, 4)
        current_bet = bb

    for token in actions_raw:
        if not isinstance(token, str):
            continue
        parts = token.split()
        if not parts:
            continue

        if parts[0] == "d":
            if len(parts) >= 2 and parts[1] == "db":
                if current_street == "preflop":
                    current_street = "flop"
                elif current_street == "flop":
                    current_street = "turn"
                elif current_street == "turn":
                    current_street = "river"
                current_bet = 0.0
                for seat in list(contrib_by_seat):
                    contrib_by_seat[seat] = 0.0
            continue

        if not parts[0].startswith("p") or len(parts) < 2:
            continue
        pos = _to_int(parts[0][1:], 0)
        seat = seat_by_pos.get(pos)
        if seat is None:
            continue
        code = parts[1].lower()

        if code == "f":
            append_action(seat, "fold", 0.0, None, None)
            continue
        if code == "sm":
            showdown_seen = True
            continue
        if code == "cc":
            already = contrib_by_seat.get(seat, 0.0)
            to_call = max(0.0, round(current_bet - already, 4))
            if to_call <= 0:
                append_action(seat, "check", 0.0, None, None)
            else:
                append_action(seat, "call", to_call, None, current_bet)
                contrib_by_seat[seat] = round(already + to_call, 4)
            continue
        if code == "cbr" and len(parts) >= 3:
            target = _to_float(parts[2], 0.0)
            already = contrib_by_seat.get(seat, 0.0)
            inc = max(0.0, round(target - already, 4))
            if current_bet <= 0:
                append_action(seat, "bet", inc, None, None)
            else:
                append_action(seat, "raise", inc, target, None)
            contrib_by_seat[seat] = round(target, 4)
            current_bet = max(current_bet, target)

    streets = _board_streets(actions_raw)

    payouts: Dict[str, float] = {}
    winners: List[str] = []
    for i, w in enumerate(winnings_raw, start=1):
        amt = round(max(0.0, _to_float(w, 0.0)), 4)
        uid = uid_by_pos.get(i)
        if uid and amt > 0:
            payouts[uid] = amt
            winners.append(uid)
    total_pot = round(sum(payouts.values()), 4) or round(pot, 4)
    hand_ended_on_street = streets[-1]["street"] if streets else "preflop"

    return {
        "metadata": {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": len(players),
            "hero_seat": 0,
            "hand_ended_on_street": hand_ended_on_street,
            "button_seat": 0,
            "sb": round(sb, 4),
            "bb": round(bb, 4),
            "ante": 0.0,
            "rng_seed_commitment": None,
        },
        "players": players,
        "streets": streets,
        "actions": actions,
        "outcome": {
            "winners": winners,
            "payouts": payouts,
            "total_pot": total_pot,
            "rake": 0.0,
            "result_reason": "showdown" if showdown_seen else ("showdown" if streets else "fold"),
            "showdown": bool(showdown_seen or streets),
        },
        "label": "bot",
    }


def _quality_ok(hand: Dict[str, Any], min_players: int, max_players: int, min_actions: int) -> bool:
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    if len(players) < min_players or len(players) > max_players:
        return False
    if len(actions) < min_actions:
        return False
    if _to_float(meta.get("bb"), 0.0) <= 0:
        return False
    for a in actions:
        before = _to_float(a.get("pot_before"), 0.0)
        after = _to_float(a.get("pot_after"), 0.0)
        if before < 0 or after < before:
            return False
    return True


def _record_hash(hand: Dict[str, Any]) -> str:
    payload = json.dumps(hand, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _infer_schema_keys(sample_json: Path) -> Tuple[set[str], Dict[str, set[str]]]:
    with sample_json.open("r", encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)
        row = json.load(f)[0] if first_non_ws == "[" else json.loads(next(ln for ln in f if ln.strip()))
    top = set(row.keys())
    nested = {
        "metadata": set((row.get("metadata") or {}).keys()),
        "outcome": set((row.get("outcome") or {}).keys()),
        "players": set((row.get("players") or [{}])[0].keys()) if row.get("players") else set(),
        "actions": set((row.get("actions") or [{}])[0].keys()) if row.get("actions") else set(),
        "streets": set((row.get("streets") or [{}])[0].keys()) if row.get("streets") else {"street", "board_cards"},
    }
    return top, nested


def _enforce_schema(hand: Dict[str, Any], top_keys: set[str], nested: Dict[str, set[str]]) -> Dict[str, Any]:
    out = {k: hand.get(k) for k in top_keys}
    md = out.get("metadata") or {}
    out["metadata"] = {k: md.get(k) for k in nested["metadata"]}
    oc = out.get("outcome") or {}
    out["outcome"] = {k: oc.get(k) for k in nested["outcome"]}
    out["players"] = [{k: p.get(k) for k in nested["players"]} for p in (out.get("players") or [])]
    out["actions"] = [{k: a.get(k) for k in nested["actions"]} for a in (out.get("actions") or [])]
    out["streets"] = [{k: s.get(k) for k in nested["streets"]} for s in (out.get("streets") or [])]
    return out


def _iter_top_level_dirs(input_dir: Path) -> List[Path]:
    dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if dirs:
        return dirs
    if list(input_dir.glob(_PHHS_GLOB)):
        return [input_dir]
    return []


def _write_parquet_from_jsonl(parquet_path: Path, jsonl_path: Path, batch_size: int) -> bool:
    if pa is None or pq is None:
        return False
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    batch: List[Dict[str, Any]] = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                batch.append(json.loads(ln))
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(str(parquet_path), table.schema, compression="snappy")
                    writer.write_table(table)
                    batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(str(parquet_path), table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return True


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-hands-per-folder", type=int, default=50000)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-every-blocks",
        type=int,
        default=100000,
        help="Print progress every N parsed PHHS blocks (0 disables).",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=5000)
    parser.add_argument("--write-parquet", action="store_true")
    parsed = parser.parse_args()
    return Args(
        sample_json=parsed.sample_json,
        input_dir=parsed.input_dir,
        out_dir=parsed.out_dir,
        max_hands_per_folder=parsed.max_hands_per_folder,
        min_players=parsed.min_players,
        max_players=parsed.max_players,
        min_actions=parsed.min_actions,
        seed=parsed.seed,
        progress_every_blocks=parsed.progress_every_blocks,
        parquet_batch_size=parsed.parquet_batch_size,
        write_parquet=parsed.write_parquet,
    )


def main() -> None:
    args = _parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir not found: {args.input_dir}")
    if args.max_hands_per_folder <= 0:
        raise SystemExit("--max-hands-per-folder must be > 0")
    if args.min_players < 2:
        raise SystemExit("--min-players must be >= 2")
    if args.max_players < args.min_players:
        raise SystemExit("--max-players must be >= --min-players")

    rng = random.Random(args.seed)
    top_keys, nested = _infer_schema_keys(args.sample_json)
    folders = _iter_top_level_dirs(args.input_dir)
    if not folders:
        raise SystemExit(f"No folders under {args.input_dir}")

    out_jsonl = args.out_dir / "poker_hands_bot_train.jsonl"
    out_parquet = args.out_dir / "poker_hands_bot_train.parquet"
    out_report = args.out_dir / "conversion_report_bot.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    global_hashes: set[str] = set()
    report: Dict[str, Any] = {"folders": {}, "total": {}}
    rows_written = 0

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for folder in folders:
            phhs_files = sorted(folder.rglob(_PHHS_GLOB))
            sample_rows: List[Tuple[str, Dict[str, Any]]] = []
            seen_in_folder = 0
            counters = {
                "phhs_files": len(phhs_files),
                "blocks_total": 0,
                "blocks_parsed": 0,
                "blocks_quality_pass": 0,
                "duplicates_dropped": 0,
                "kept": 0,
            }
            print(
                f"[{folder.name}] scanning {len(phhs_files)} .phhs file(s) "
                f"(target={args.max_hands_per_folder}, players={args.min_players}..{args.max_players})",
                flush=True,
            )
            for phhs_path in phhs_files:
                for block in _iter_phhs_blocks(phhs_path):
                    counters["blocks_total"] += 1
                    pe = int(args.progress_every_blocks)
                    if pe > 0 and counters["blocks_total"] % pe == 0:
                        print(
                            f"[{folder.name}] blocks={counters['blocks_total']} "
                            f"parsed={counters['blocks_parsed']} quality={counters['blocks_quality_pass']} "
                            f"kept_so_far={len(sample_rows)}",
                            flush=True,
                        )
                    hand = _canonicalize_hand(block)
                    if hand is None:
                        continue
                    counters["blocks_parsed"] += 1
                    if not _quality_ok(
                        hand,
                        min_players=args.min_players,
                        max_players=args.max_players,
                        min_actions=args.min_actions,
                    ):
                        continue
                    hand = _enforce_schema(hand, top_keys, nested)
                    counters["blocks_quality_pass"] += 1
                    h = _record_hash(hand)
                    if h in global_hashes:
                        counters["duplicates_dropped"] += 1
                        continue
                    seen_in_folder += 1
                    if len(sample_rows) < args.max_hands_per_folder:
                        sample_rows.append((h, hand))
                    else:
                        j = rng.randint(1, seen_in_folder)
                        if j <= args.max_hands_per_folder:
                            sample_rows[j - 1] = (h, hand)

            for h, row in sample_rows:
                out_f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                global_hashes.add(h)
            counters["kept"] = len(sample_rows)
            rows_written += len(sample_rows)
            report["folders"][folder.name] = counters
            print(
                f"[{folder.name}] kept {len(sample_rows)} bot hands "
                f"(players {args.min_players}..{args.max_players})"
            )

    parquet_ok = False
    if args.write_parquet:
        parquet_ok = _write_parquet_from_jsonl(out_parquet, out_jsonl, args.parquet_batch_size)

    report["total"] = {
        "rows_written": rows_written,
        "seed": args.seed,
        "max_hands_per_folder": args.max_hands_per_folder,
        "min_players": args.min_players,
        "max_players": args.max_players,
        "min_actions": args.min_actions,
        "label": "bot",
        "hu_only": args.min_players == 2 and args.max_players == 2,
        "parquet_written": parquet_ok,
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote JSONL: {out_jsonl}")
    if args.write_parquet and parquet_ok:
        print(f"Wrote Parquet: {out_parquet}")
    elif args.write_parquet:
        print("Parquet requested but pyarrow is unavailable.")
    print(f"Wrote report: {out_report}")
    print(f"Total kept rows: {rows_written}")


if __name__ == "__main__":
    main()
