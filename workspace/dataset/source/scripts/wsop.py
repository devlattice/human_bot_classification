#!/usr/bin/env python3
"""Convert WSOP .phh files into canonical Poker44 JSONL for stress testing.


python workspace/dataset/source/scripts/wsop.py \
  --wsop-dir workspace/dataset/source/zenodo/wsop \
  --sample-json workspace/dataset/source/data/poker_hands_train.json \
  --out-dir workspace/dataset/source/data/wsop_v1 \
  --label human \
  --min-players 2 \
  --min-actions 6 \
  --progress-every 5 \
  --require-showdown

"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


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


def _stable_uid(raw_player: str) -> str:
    digest = hashlib.sha256(raw_player.encode("utf-8")).hexdigest()
    return f"p_{digest}"


def _parse_rhs(raw: str) -> Any:
    value = raw.strip()
    if _TIME_RE.match(value):
        return value
    value = value.replace("true", "True").replace("false", "False").replace("null", "None")
    return ast.literal_eval(value)


def _parse_phh_file(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, rhs = line.split("=", 1)
            try:
                out[key.strip()] = _parse_rhs(rhs.strip())
            except Exception:
                continue
    return out


def _iter_phh_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.rglob("*.phh"))


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
            elif split_cards:
                board_cards = board_cards + [split_cards[0]]
                if idx < len(labels):
                    streets.append({"street": labels[idx], "board_cards": board_cards.copy()})
                    idx += 1
    return streets


def _variant_to_metadata(variant: str, bb: float, n_players: int) -> Tuple[str, str]:
    v = (variant or "").upper()
    if v == "NT":
        return "Hold'em", "No Limit"
    if v == "PO":
        return "Omaha", "Pot Limit"
    if v in {"F7S", "R7S", "S8B", "FR"}:
        return "Mixed", "Fixed Limit"
    return "Mixed", "No Limit" if bb > 0 else ""


def _canonicalize_wsop_hand(raw: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    players_raw = raw.get("players") or []
    actions_raw = raw.get("actions") or []
    stacks_raw = raw.get("starting_stacks") or []
    winnings_raw = raw.get("finishing_stacks") or raw.get("winnings") or []
    antes = raw.get("antes") or []
    blinds = raw.get("blinds_or_straddles") or []
    bring_in = _to_float(raw.get("bring_in"), 0.0)
    small_bet = _to_float(raw.get("small_bet"), 0.0)
    big_bet = _to_float(raw.get("big_bet"), 0.0)
    seats_raw = raw.get("seats") or list(range(1, len(players_raw) + 1))

    if not isinstance(players_raw, list) or not isinstance(actions_raw, list):
        return None
    if len(players_raw) < 2:
        return None

    sb = _to_float(blinds[0], 0.0) if len(blinds) > 0 else 0.0
    bb = _to_float(blinds[1], 0.0) if len(blinds) > 1 else 0.0
    if bb <= 0:
        bb = big_bet if big_bet > 0 else (small_bet if small_bet > 0 else (bring_in if bring_in > 0 else 1.0))
    ante = max(_to_float(antes[0], 0.0) if isinstance(antes, list) and antes else 0.0, bring_in if bring_in > 0 else 0.0)

    players: List[Dict[str, Any]] = []
    uid_by_pos: Dict[int, str] = {}
    seat_by_pos: Dict[int, int] = {}
    for i, p in enumerate(players_raw, start=1):
        seat = _to_int(seats_raw[i - 1], i) if i - 1 < len(seats_raw) else i
        uid = _stable_uid(str(p))
        cards = None
        showed = False
        players.append(
            {
                "player_uid": uid,
                "seat": seat,
                "starting_stack": round(_to_float(stacks_raw[i - 1], 0.0) if i - 1 < len(stacks_raw) else 0.0, 4),
                "hole_cards": cards,
                "showed_hand": showed,
            }
        )
        uid_by_pos[i] = uid
        seat_by_pos[i] = seat

    actions: List[Dict[str, Any]] = []
    pot = 0.0
    action_id = 1
    street = "preflop"
    current_bet = 0.0
    contrib_by_seat: Dict[int, float] = {p["seat"]: 0.0 for p in players}
    shown_cards: Dict[int, Sequence[str]] = {}
    showdown_seen = False

    def _emit(actor_seat: int, action_type: str, amount: float, raise_to: Optional[float], call_to: Optional[float]) -> None:
        nonlocal pot, action_id
        amount = max(0.0, round(amount, 4))
        before = round(pot, 4)
        after = round(before + amount, 4)
        actions.append(
            {
                "action_id": str(action_id),
                "street": street,
                "actor_seat": actor_seat,
                "action_type": action_type,
                "amount": amount,
                "raise_to": None if raise_to is None else round(raise_to, 4),
                "call_to": None if call_to is None else round(call_to, 4),
                "normalized_amount_bb": round(amount / bb, 4) if bb > 0 else 0.0,
                "pot_before": before,
                "pot_after": after,
            }
        )
        pot = after
        action_id += 1

    for tok in actions_raw:
        if not isinstance(tok, str):
            continue
        parts = tok.split()
        if not parts:
            continue
        if parts[0] == "d":
            if len(parts) >= 2 and parts[1] == "db":
                if street == "preflop":
                    street = "flop"
                elif street == "flop":
                    street = "turn"
                elif street == "turn":
                    street = "river"
                current_bet = 0.0
                for k in list(contrib_by_seat):
                    contrib_by_seat[k] = 0.0
            elif len(parts) >= 4 and parts[1] == "dh":
                pos = _to_int(parts[2].lstrip("p"), 0)
                if pos > 0 and parts[3] != "????":
                    cards = [parts[3][i : i + 2] for i in range(0, len(parts[3]), 2)]
                    shown_cards[pos] = cards
            continue

        if not parts[0].startswith("p") or len(parts) < 2:
            continue
        pos = _to_int(parts[0][1:], 0)
        seat = seat_by_pos.get(pos)
        if seat is None:
            continue
        code = parts[1].lower()

        if code == "f":
            _emit(seat, "fold", 0.0, None, None)
        elif code == "cc":
            already = contrib_by_seat.get(seat, 0.0)
            to_call = max(0.0, round(current_bet - already, 4))
            if to_call > 0:
                _emit(seat, "call", to_call, None, current_bet)
                contrib_by_seat[seat] = round(already + to_call, 4)
            else:
                _emit(seat, "check", 0.0, None, None)
        elif code == "cbr" and len(parts) >= 3:
            tgt = _to_float(parts[2], 0.0)
            already = contrib_by_seat.get(seat, 0.0)
            inc = max(0.0, round(tgt - already, 4))
            if current_bet <= 0:
                _emit(seat, "bet", inc, None, None)
            else:
                _emit(seat, "raise", inc, tgt, None)
            contrib_by_seat[seat] = round(tgt, 4)
            current_bet = max(current_bet, tgt)
        elif code == "pb":
            _emit(seat, "ante", ante, None, None)
            contrib_by_seat[seat] = round(contrib_by_seat.get(seat, 0.0) + ante, 4)
            current_bet = max(current_bet, ante)
        elif code == "sm":
            showdown_seen = True
            if len(parts) >= 3:
                cards = [parts[2][i : i + 2] for i in range(0, len(parts[2]), 2)]
                shown_cards[pos] = cards

    for pos, cards in shown_cards.items():
        seat = seat_by_pos.get(pos)
        if seat is None:
            continue
        for p in players:
            if p["seat"] == seat:
                p["hole_cards"] = list(cards)
                p["showed_hand"] = True
                break

    payouts: Dict[str, float] = {}
    winners: List[str] = []
    if isinstance(winnings_raw, list) and len(winnings_raw) == len(players_raw) and raw.get("finishing_stacks"):
        # finishing_stacks: infer net win/loss from start -> finish.
        for i, st in enumerate(stacks_raw, start=1):
            start = _to_float(st, 0.0)
            finish = _to_float(winnings_raw[i - 1], 0.0)
            delta = round(finish - start, 4)
            if delta > 0:
                uid = uid_by_pos[i]
                payouts[uid] = delta
                winners.append(uid)
    elif isinstance(winnings_raw, list):
        for i, w in enumerate(winnings_raw, start=1):
            amt = round(max(0.0, _to_float(w, 0.0)), 4)
            if amt > 0:
                uid = uid_by_pos.get(i)
                if uid:
                    payouts[uid] = amt
                    winners.append(uid)

    total_pot = round(sum(payouts.values()), 4)
    if total_pot <= 0:
        total_pot = round(pot, 4)

    streets = _board_streets(actions_raw)
    variant = str(raw.get("variant") or "")
    game_type, limit_type = _variant_to_metadata(variant, bb=bb, n_players=len(players))
    hand_ended_on_street = streets[-1]["street"] if streets else "preflop"

    return {
        "metadata": {
            "game_type": game_type,
            "limit_type": limit_type,
            "max_seats": len(players),
            "hero_seat": 0,
            "hand_ended_on_street": hand_ended_on_street,
            "button_seat": 0,
            "sb": round(sb, 4),
            "bb": round(bb, 4),
            "ante": round(ante, 4),
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


def _quality_ok(hand: Dict[str, Any], min_players: int, min_actions: int, require_showdown: bool) -> bool:
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    outcome = hand.get("outcome") or {}
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
    if require_showdown and not bool(outcome.get("showdown")):
        return False
    if not any(bool(p.get("showed_hand")) for p in players):
        return False
    return True


def _record_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wsop-dir", type=Path, required=True, help="Root directory containing WSOP .phh files.")
    p.add_argument("--sample-json", type=Path, required=True, help="Reference canonical sample schema JSON/JSONL.")
    p.add_argument("--out-dir", type=Path, default=here.parent / "data" / "wsop_v1", help="Output directory for JSONL + report.")
    p.add_argument("--out-name", type=str, default="wsop_hands_stress.jsonl", help="Output JSONL filename.")
    p.add_argument("--label", type=str, default="human", choices=["human", "bot"], help="Label set on output records.")
    p.add_argument("--min-players", type=int, default=2, help="Drop hands with fewer players.")
    p.add_argument("--min-actions", type=int, default=6, help="Drop hands with fewer actions.")
    p.add_argument("--require-showdown", action="store_true", help="Keep only showdown hands (higher reliability).")
    p.add_argument("--progress-every", type=int, default=500, help="Print progress every N files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wsop_dir = args.wsop_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_path = out_dir / args.out_name
    report_path = out_dir / "wsop_export_report.json"

    if not wsop_dir.is_dir():
        raise SystemExit(f"--wsop-dir not found: {wsop_dir}")
    top_keys, nested = _infer_schema_keys(args.sample_json.expanduser().resolve())

    phh_files = list(_iter_phh_files(wsop_dir))
    if not phh_files:
        raise SystemExit(f"No .phh files found under {wsop_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    dropped_parse = 0
    dropped_quality = 0
    dropped_dupe = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, phh in enumerate(phh_files, start=1):
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[progress] files={i}/{len(phh_files)} kept={kept}", flush=True)
            raw = _parse_phh_file(phh)
            if not raw:
                dropped_parse += 1
                continue
            hand = _canonicalize_wsop_hand(raw, label=args.label)
            if hand is None:
                dropped_parse += 1
                continue
            hand = _enforce_schema(hand, top_keys=top_keys, nested=nested)
            if not _quality_ok(hand, args.min_players, args.min_actions, args.require_showdown):
                dropped_quality += 1
                continue
            if not _row_schema_matches(hand, top_keys=top_keys, nested=nested):
                raise SystemExit(f"Schema mismatch for file {phh}")
            h = _record_hash(hand)
            if h in seen:
                dropped_dupe += 1
                continue
            seen.add(h)
            out_f.write(json.dumps(hand, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1

    report = {
        "wsop_dir": str(wsop_dir),
        "sample_json": str(args.sample_json),
        "out_jsonl": str(out_path),
        "files_total": len(phh_files),
        "rows_written": kept,
        "dropped_parse": dropped_parse,
        "dropped_quality": dropped_quality,
        "dropped_duplicates": dropped_dupe,
        "label": args.label,
        "min_players": args.min_players,
        "min_actions": args.min_actions,
        "require_showdown": bool(args.require_showdown),
        "schema_exact_match": True,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote JSONL: {out_path}")
    print(f"Wrote report: {report_path}")
    print(f"Rows written: {kept}")
    print("Schema check (all rows): PASS")


if __name__ == "__main__":
    main()
