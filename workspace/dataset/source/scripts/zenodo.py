#!/usr/bin/env python3
"""Convert Zenodo PHHS handhq dumps into Poker44 canonical hand schema.

Key properties:
- Streams PHHS files (works with very large corpora).
- Loops every first-level folder inside --input-dir.
- Keeps at most N high-quality hands per folder (default 10,000).
- Removes duplicates globally.
- Emits canonical JSONL and Parquet (if pyarrow is installed).


python workspace/dataset/source/scripts/zenodo.py \
  --sample-json workspace/dataset/source/data/poker_hands_train.json \
  --input-dir workspace/dataset/source/zenodo/handhq \
  --out-dir workspace/dataset/source/data/zenodo_v3 \
  --max-hands-per-folder 45000 \
  --min-players 2 \
  --min-actions 4 \
  --min-file-quality-rate 0.4 \
  --min-file-quality-hands 50 \
  --randomize-files \
  --seed 42
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
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
    min_actions: int
    seed: int
    label: str
    parquet_batch_size: int
    progress_every: int
    write_parquet: bool
    randomize_files: bool
    min_file_quality_rate: float
    min_file_quality_hands: int


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
            key = key.strip()
            rhs = rhs.strip()
            try:
                current[key] = _parse_rhs(rhs)
            except Exception:
                # Skip malformed values but continue parsing file.
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
        if len(parts) >= 3 and parts[0] == "d" and parts[1] == "db":
            cards = parts[2]
            if len(cards) % 2 != 0:
                continue
            split_cards = [cards[i : i + 2] for i in range(0, len(cards), 2)]
            if idx == 0 and len(split_cards) >= 3:
                board_cards = split_cards[:3]
                streets.append({"street": labels[idx], "board_cards": board_cards.copy()})
                idx += 1
            else:
                if not split_cards:
                    continue
                board_cards = board_cards + [split_cards[0]]
                if idx < len(labels):
                    streets.append({"street": labels[idx], "board_cards": board_cards.copy()})
                    idx += 1
    return streets


def _canonicalize_hand(block: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
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
        *,
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

    # Convert posted blinds to explicit canonical actions.
    if len(players) >= 1 and sb > 0:
        seat = seat_by_pos.get(1, 1)
        append_action(actor_seat=seat, action_type="small_blind", amount=sb, raise_to=None, call_to=None)
        contrib_by_seat[seat] = round(contrib_by_seat.get(seat, 0.0) + sb, 4)
    if len(players) >= 2 and bb > 0:
        seat = seat_by_pos.get(2, 2)
        append_action(actor_seat=seat, action_type="big_blind", amount=bb, raise_to=None, call_to=None)
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
            append_action(actor_seat=seat, action_type="fold", amount=0.0, raise_to=None, call_to=None)
            continue
        if code == "sm":
            showdown_seen = True
            continue
        if code == "cc":
            already = contrib_by_seat.get(seat, 0.0)
            to_call = max(0.0, round(current_bet - already, 4))
            if to_call <= 0:
                append_action(actor_seat=seat, action_type="check", amount=0.0, raise_to=None, call_to=None)
            else:
                append_action(actor_seat=seat, action_type="call", amount=to_call, raise_to=None, call_to=current_bet)
                contrib_by_seat[seat] = round(already + to_call, 4)
            continue
        if code == "cbr":
            if len(parts) < 3:
                continue
            target = _to_float(parts[2], 0.0)
            already = contrib_by_seat.get(seat, 0.0)
            inc = max(0.0, round(target - already, 4))
            if current_bet <= 0:
                append_action(actor_seat=seat, action_type="bet", amount=inc, raise_to=None, call_to=None)
            else:
                append_action(actor_seat=seat, action_type="raise", amount=inc, raise_to=target, call_to=None)
            contrib_by_seat[seat] = round(target, 4)
            current_bet = max(current_bet, target)
            continue

    streets = _board_streets(actions_raw)

    payouts: Dict[str, float] = {}
    winners: List[str] = []
    for i, w in enumerate(winnings_raw, start=1):
        amt = round(max(0.0, _to_float(w, 0.0)), 4)
        uid = uid_by_pos.get(i)
        if not uid:
            continue
        if amt > 0:
            payouts[uid] = amt
            winners.append(uid)
    total_pot = round(sum(payouts.values()), 4)
    if total_pot <= 0:
        total_pot = round(pot, 4)

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
        "label": label,
    }


def _quality_ok(hand: Dict[str, Any], min_players: int, min_actions: int) -> bool:
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    if len(players) < min_players:
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
    nested = {}
    for k in ("metadata", "outcome"):
        nested[k] = set((row.get(k) or {}).keys())
    nested["players"] = set((row.get("players") or [{}])[0].keys()) if row.get("players") else set()
    nested["actions"] = set((row.get("actions") or [{}])[0].keys()) if row.get("actions") else set()
    nested["streets"] = set((row.get("streets") or [{}])[0].keys()) if row.get("streets") else {"street", "board_cards"}
    return top, nested


def _enforce_schema(hand: Dict[str, Any], top_keys: set[str], nested: Dict[str, set[str]]) -> Dict[str, Any]:
    out = {k: hand.get(k) for k in top_keys}

    md = out.get("metadata") or {}
    out["metadata"] = {k: md.get(k) for k in nested["metadata"]}

    oc = out.get("outcome") or {}
    out["outcome"] = {k: oc.get(k) for k in nested["outcome"]}

    players = out.get("players") or []
    out["players"] = [{k: p.get(k) for k in nested["players"]} for p in players]

    actions = out.get("actions") or []
    out["actions"] = [{k: a.get(k) for k in nested["actions"]} for a in actions]

    streets = out.get("streets") or []
    out["streets"] = [{k: s.get(k) for k in nested["streets"]} for s in streets]

    return out


def _row_schema_matches(row: Dict[str, Any], top_keys: set[str], nested: Dict[str, set[str]]) -> bool:
    if set(row.keys()) != top_keys:
        return False
    if set((row.get("metadata") or {}).keys()) != nested["metadata"]:
        return False
    if set((row.get("outcome") or {}).keys()) != nested["outcome"]:
        return False

    for p in (row.get("players") or []):
        if set((p or {}).keys()) != nested["players"]:
            return False
    for a in (row.get("actions") or []):
        if set((a or {}).keys()) != nested["actions"]:
            return False
    for s in (row.get("streets") or []):
        if set((s or {}).keys()) != nested["streets"]:
            return False
    return True


def _reservoir_sample(rng: random.Random, sample: List[Dict[str, Any]], item: Dict[str, Any], seen: int, limit: int) -> None:
    if len(sample) < limit:
        sample.append(item)
        return
    j = rng.randint(1, seen)
    if j <= limit:
        sample[j - 1] = item


def _stratum_key(hand: Dict[str, Any]) -> str:
    bb = _to_float((hand.get("metadata") or {}).get("bb"), 0.0)
    n_players = len(hand.get("players") or [])
    n_streets = len(hand.get("streets") or [])
    n_actions = len(hand.get("actions") or [])

    if bb < 0.1:
        stake = "micro"
    elif bb < 1.0:
        stake = "low"
    elif bb < 5.0:
        stake = "mid"
    else:
        stake = "high"

    if n_players <= 2:
        seats = "hu"
    elif n_players <= 6:
        seats = "sh"
    else:
        seats = "fr"

    if n_streets == 0:
        depth = "pf"
    elif n_streets == 1:
        depth = "flop"
    else:
        depth = "deep"

    if n_actions <= 6:
        pace = "short"
    elif n_actions <= 14:
        pace = "mid"
    else:
        pace = "long"

    return f"{stake}|{seats}|{depth}|{pace}"


def _build_quotas(stratum_counts: Dict[str, int], target: int) -> Dict[str, int]:
    total = sum(stratum_counts.values())
    if total <= 0 or target <= 0:
        return {}
    raw = {k: (v / total) * target for k, v in stratum_counts.items() if v > 0}
    quotas = {k: min(stratum_counts[k], int(math.floor(x))) for k, x in raw.items()}
    used = sum(quotas.values())
    remaining = target - used
    if remaining > 0:
        rank = sorted(raw.items(), key=lambda kv: (kv[1] - math.floor(kv[1])), reverse=True)
        for key, _ in rank:
            if remaining <= 0:
                break
            if quotas[key] < stratum_counts[key]:
                quotas[key] += 1
                remaining -= 1
    return quotas


def _profile_folder(
    folder: Path,
    phhs_files: List[Path],
    args: Args,
    top_keys: set[str],
    nested: Dict[str, set[str]],
) -> Tuple[Dict[str, Any], List[Path], Dict[str, int]]:
    counters = {
        "phhs_files": len(phhs_files),
        "blocks_total": 0,
        "blocks_parsed": 0,
        "blocks_quality_pass": 0,
        "eligible_files": 0,
    }
    file_stats: Dict[str, Dict[str, int]] = {}
    stratum_counts: Dict[str, int] = {}
    eligible_files: List[Path] = []

    for file_idx, phhs_path in enumerate(phhs_files, start=1):
        if args.progress_every > 0 and file_idx % 50 == 0:
            print(f"[{folder.name}] profiling file {file_idx}/{len(phhs_files)}", flush=True)
        st = {"blocks_total": 0, "blocks_parsed": 0, "quality_pass": 0}
        for block in _iter_phhs_blocks(phhs_path):
            counters["blocks_total"] += 1
            st["blocks_total"] += 1
            hand = _canonicalize_hand(block, label=args.label)
            if hand is None:
                continue
            counters["blocks_parsed"] += 1
            st["blocks_parsed"] += 1
            if not _quality_ok(hand, min_players=args.min_players, min_actions=args.min_actions):
                continue
            hand = _enforce_schema(hand, top_keys=top_keys, nested=nested)
            counters["blocks_quality_pass"] += 1
            st["quality_pass"] += 1
            key = _stratum_key(hand)
            stratum_counts[key] = stratum_counts.get(key, 0) + 1

        file_stats[phhs_path.name] = st
        ratio = (st["quality_pass"] / st["blocks_total"]) if st["blocks_total"] else 0.0
        if st["quality_pass"] >= args.min_file_quality_hands and ratio >= args.min_file_quality_rate:
            eligible_files.append(phhs_path)

    counters["eligible_files"] = len(eligible_files)
    return {"counters": counters, "file_stats": file_stats}, eligible_files, stratum_counts


def _iter_top_level_dirs(input_dir: Path) -> List[Path]:
    dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if dirs:
        return dirs
    # Fallback: allow input-dir to be a single folder containing .phhs files.
    if list(input_dir.glob(_PHHS_GLOB)):
        return [input_dir]
    return []


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_parquet(path: Path, rows: Sequence[Dict[str, Any]], batch_size: int) -> bool:
    if pa is None or pq is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[pq.ParquetWriter] = None
    try:
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            table = pa.Table.from_pylist(batch_rows)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return True


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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json", type=Path, required=True, help="Reference canonical JSON file to copy schema from.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing many PHHS source folders.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSONL + Parquet + report.")
    parser.add_argument("--max-hands-per-folder", type=int, default=10000, help="Reservoir sample cap per top-level folder.")
    parser.add_argument("--min-players", type=int, default=2, help="Drop hand if player count is lower than this.")
    parser.add_argument("--min-actions", type=int, default=4, help="Drop hand if action count is lower than this.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    parser.add_argument("--label", type=str, default="human", choices=["human", "bot"], help="Label assigned to converted records.")
    parser.add_argument("--parquet-batch-size", type=int, default=5000, help="Rows per parquet write batch.")
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress every N parsed PHHS blocks (0 disables).")
    parser.add_argument("--write-parquet", action="store_true", help="Also write parquet output. Disabled by default.")
    parser.add_argument("--randomize-files", action="store_true", help="Shuffle .phhs files in each folder before parsing (recommended).")
    parser.add_argument("--min-file-quality-rate", type=float, default=0.4, help="Drop .phhs files whose quality-pass ratio is below this threshold.")
    parser.add_argument("--min-file-quality-hands", type=int, default=50, help="Drop .phhs files with fewer quality hands than this threshold.")
    parsed = parser.parse_args()
    return Args(
        sample_json=parsed.sample_json,
        input_dir=parsed.input_dir,
        out_dir=parsed.out_dir,
        max_hands_per_folder=parsed.max_hands_per_folder,
        min_players=parsed.min_players,
        min_actions=parsed.min_actions,
        seed=parsed.seed,
        label=parsed.label,
        parquet_batch_size=parsed.parquet_batch_size,
        progress_every=parsed.progress_every,
        write_parquet=parsed.write_parquet,
        randomize_files=parsed.randomize_files,
        min_file_quality_rate=parsed.min_file_quality_rate,
        min_file_quality_hands=parsed.min_file_quality_hands,
    )


def main() -> None:
    args = _parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"--input-dir not found: {args.input_dir}")
    if args.max_hands_per_folder <= 0:
        raise SystemExit("--max-hands-per-folder must be > 0")

    rng = random.Random(args.seed)
    top_keys, nested = _infer_schema_keys(args.sample_json)
    folder_dirs = _iter_top_level_dirs(args.input_dir)
    if not folder_dirs:
        raise SystemExit(f"No folders under {args.input_dir}")

    global_seen_hashes: set[str] = set()
    report: Dict[str, Any] = {"folders": {}, "total": {}}
    out_jsonl = args.out_dir / "poker_hands_zenodo_v3.jsonl"
    out_parquet = args.out_dir / "poker_hands_zenodo_v3.parquet"
    out_report = args.out_dir / "conversion_report.json"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_rows_written = 0

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for folder in folder_dirs:
            folder_name = folder.name
            counters = {
                "phhs_files": 0,
                "phhs_files_processed": 0,
                "blocks_total": 0,
                "blocks_parsed": 0,
                "blocks_quality_pass": 0,
                "blocks_schema_kept": 0,
                "duplicates_dropped": 0,
                "eligible_files": 0,
            }
            phhs_files = sorted(folder.rglob(_PHHS_GLOB))
            if args.randomize_files:
                rng.shuffle(phhs_files)
            counters["phhs_files"] = len(phhs_files)
            print(f"[{folder_name}] profiling {len(phhs_files)} phhs files...", flush=True)
            profile, eligible_files, stratum_counts = _profile_folder(
                folder=folder,
                phhs_files=phhs_files,
                args=args,
                top_keys=top_keys,
                nested=nested,
            )
            counters.update(profile["counters"])
            if args.randomize_files:
                rng.shuffle(eligible_files)

            target = args.max_hands_per_folder
            quotas = _build_quotas(stratum_counts, target)
            sample_limit = max(target, int(target * 1.35))
            sampled_by_stratum: Dict[str, List[Dict[str, Any]]] = {k: [] for k in quotas}
            seen_by_stratum: Dict[str, int] = {k: 0 for k in quotas}
            fallback_rows: List[Dict[str, Any]] = []
            fallback_seen = 0
            local_seen_hashes: set[str] = set()

            print(
                f"[{folder_name}] sampling from {len(eligible_files)} eligible files "
                f"with {len(quotas)} strata (target={target})",
                flush=True,
            )

            for file_idx, phhs_path in enumerate(eligible_files, start=1):
                counters["phhs_files_processed"] += 1
                if file_idx % 25 == 0:
                    print(f"[{folder_name}] sampling file {file_idx}/{len(eligible_files)}", flush=True)
                for block in _iter_phhs_blocks(phhs_path):
                    hand = _canonicalize_hand(block, label=args.label)
                    if hand is None:
                        continue
                    if not _quality_ok(hand, min_players=args.min_players, min_actions=args.min_actions):
                        continue
                    hand = _enforce_schema(hand, top_keys=top_keys, nested=nested)
                    h = _record_hash(hand)
                    if h in global_seen_hashes or h in local_seen_hashes:
                        counters["duplicates_dropped"] += 1
                        continue
                    local_seen_hashes.add(h)
                    key = _stratum_key(hand)
                    if key in quotas and quotas[key] > 0:
                        seen_by_stratum[key] += 1
                        _reservoir_sample(
                            rng=rng,
                            sample=sampled_by_stratum[key],
                            item=hand,
                            seen=seen_by_stratum[key],
                            limit=quotas[key],
                        )
                    fallback_seen += 1
                    _reservoir_sample(
                        rng=rng,
                        sample=fallback_rows,
                        item=hand,
                        seen=fallback_seen,
                        limit=sample_limit,
                    )

            selected_rows: List[Dict[str, Any]] = []
            selected_hashes: set[str] = set()
            for rows in sampled_by_stratum.values():
                for row in rows:
                    h = _record_hash(row)
                    if h in selected_hashes:
                        continue
                    selected_hashes.add(h)
                    selected_rows.append(row)
            if len(selected_rows) < target:
                for row in fallback_rows:
                    h = _record_hash(row)
                    if h in selected_hashes:
                        continue
                    selected_hashes.add(h)
                    selected_rows.append(row)
                    if len(selected_rows) >= target:
                        break
            if len(selected_rows) > target:
                rng.shuffle(selected_rows)
                selected_rows = selected_rows[:target]

            for row in selected_rows:
                if not _row_schema_matches(row, top_keys=top_keys, nested=nested):
                    raise SystemExit(f"Schema mismatch detected in folder {folder_name}; aborting.")
                out_f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                global_seen_hashes.add(_record_hash(row))

            counters["blocks_schema_kept"] = len(selected_rows)
            total_rows_written += len(selected_rows)
            report["folders"][folder_name] = {
                **counters,
                "strata_profile_counts": stratum_counts,
                "strata_quota_counts": quotas,
            }
            print(
                f"[{folder_name}] kept={len(selected_rows)} unique_hands "
                f"(eligible_files={len(eligible_files)})",
                flush=True,
            )

    parquet_ok = False
    if args.write_parquet:
        parquet_ok = _write_parquet_from_jsonl(
            parquet_path=out_parquet,
            jsonl_path=out_jsonl,
            batch_size=args.parquet_batch_size,
        )
        if not parquet_ok:
            out_parquet = None
    else:
        out_parquet = None

    schema_ok = True

    report["total"] = {
        "rows_written": total_rows_written,
        "seed": args.seed,
        "max_hands_per_folder": args.max_hands_per_folder,
        "min_players": args.min_players,
        "min_actions": args.min_actions,
        "label": args.label,
        "min_file_quality_rate": args.min_file_quality_rate,
        "min_file_quality_hands": args.min_file_quality_hands,
        "parquet_written": parquet_ok,
        "schema_check_all_rows": schema_ok,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote JSONL: {out_jsonl}")
    if out_parquet is not None:
        print(f"Wrote Parquet: {out_parquet}")
    else:
        print("Skipped Parquet (enable with --write-parquet).")
    print(f"Wrote report: {out_report}")
    print(f"Total kept rows: {total_rows_written}")
    print(f"Schema check (all rows) vs sample: {'PASS' if schema_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
