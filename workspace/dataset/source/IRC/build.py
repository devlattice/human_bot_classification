#!/usr/bin/env python3
"""
Build Poker44-style normalized hands from the IRC Poker Dataset with streaming IO.

Pipeline:
1) Download/extract IRCdata.tgz (optional).
2) Parse month archives (holdem.YYYYMM.tgz) into on-disk SQLite tables (hdb + pdb).
3) Stream-join by hand_id and emit normalized hand JSONL.
4) Apply strict QC gates and write rejects with reasons.

Outputs:
- accepted JSONL (one normalized hand per line)
- rejects JSONL (raw metadata + reject reasons)
- qc_summary.json

Example:
  python workspace/dataset/source/IRC/build.py --sample 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import ssl
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.validator.chunk_features import aggregate_chunk_from_hands
from poker44.validator.sanitization import sanitize_hand_for_miner


IRC_URL = "https://poker.cs.ualberta.ca/IRC/IRCdata.tgz"
SCRIPT_VERSION = "irc_build_v1"
SCHEMA_VERSION = "poker44_hand_v1"

BASE_DIR = Path("workspace/dataset/source/IRC")
ARCHIVE_PATH = BASE_DIR / "IRCdata.tgz"
EXTRACT_DIR = BASE_DIR / "raw"
OUT_ACCEPTED = BASE_DIR / "poker_hands_irc_normalized.jsonl"
OUT_REJECTS = BASE_DIR / "poker_hands_irc_rejects.jsonl"
OUT_SUMMARY = BASE_DIR / "qc_summary.json"
TMP_DIR = BASE_DIR / "_tmp_sqlite"

CARD_RE = re.compile(r"^[2-9TJQKA][shdc]$", re.IGNORECASE)
TOKEN_RE = re.compile(r"([A-Za-z])(\d*)")


@dataclass
class HdbRecord:
    hand_id: str
    table_id: str
    month: str
    num_players: int
    board_cards: List[str]
    street_pots: Dict[str, float]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _log(f"[download] skip existing archive: {dest}")
        return
    _log(f"[download] {url} -> {dest}")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1 << 20
        got = 0
        with dest.open("wb") as f:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                got += len(data)
                if total > 0:
                    pct = (got / total) * 100.0
                    if got % (64 << 20) < chunk:
                        _log(f"[download] {got/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)")
                elif got % (64 << 20) < chunk:
                    _log(f"[download] {got/1e6:.1f}MB")
    _log("[download] done")


def _safe_extract_tar_gz(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted_ok"
    if marker.exists():
        _log(f"[extract] skip already extracted: {dest}")
        return
    _log(f"[extract] {archive} -> {dest}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    marker.write_text("ok", encoding="utf-8")
    _log("[extract] done")


def _iter_lines_from_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Iterator[str]:
    fobj = tf.extractfile(member)
    if fobj is None:
        return
    for raw in fobj:
        try:
            yield raw.decode("utf-8", errors="replace").rstrip("\n")
        except Exception:
            continue


def _norm_card(token: str) -> Optional[str]:
    t = token.strip()
    if not CARD_RE.match(t):
        return None
    return t[0].upper() + t[1].lower()


def _parse_hdb_line(line: str, month: str) -> Optional[HdbRecord]:
    # Expected: timestamp table_id hand_id num_players p/f p/f p/f p/f [board...]
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    # In IRC hdb, the first field is the unique hand timestamp used by pdb as well.
    hand_id = parts[0]
    table_id = parts[1]
    try:
        num_players = int(parts[3])
    except ValueError:
        return None

    street_names = ["preflop", "flop", "turn", "river"]
    street_pots: Dict[str, float] = {}
    for i, st in enumerate(street_names, start=4):
        field = parts[i] if i < len(parts) else ""
        if "/" not in field:
            continue
        try:
            street_pots[st] = float(field.split("/", 1)[1])
        except ValueError:
            pass

    board_cards: List[str] = []
    for tok in parts[8:]:
        c = _norm_card(tok)
        if c:
            board_cards.append(c)
    return HdbRecord(
        hand_id=hand_id,
        table_id=table_id,
        month=month,
        num_players=num_players,
        board_cards=board_cards,
        street_pots=street_pots,
    )


def _parse_action_tokens(s: str) -> List[str]:
    if not s or s == "-":
        return []
    out: List[str] = []
    for m in TOKEN_RE.finditer(s):
        code = m.group(1)
        amt = m.group(2) or ""
        if code.isalpha():
            out.append(f"{code}{amt}")
    return out


def _parse_pdb_line(line: str) -> Optional[dict]:
    # player ts hand_id seat pre flop turn river bankroll winnings unknown [hole1 hole2]
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        seat = int(parts[3])
        bankroll = float(parts[8])
        winnings = float(parts[9])
    except ValueError:
        return None
    hole = []
    if len(parts) >= 13:
        c1 = _norm_card(parts[11])
        c2 = _norm_card(parts[12])
        if c1 and c2:
            hole = [c1, c2]
    return {
        "player": parts[0],
        # In IRC pdb, hand timestamp is field[1] and joins to hdb field[0].
        "hand_id": parts[1],
        "seat": seat,
        "preflop": _parse_action_tokens(parts[4]),
        "flop": _parse_action_tokens(parts[5]),
        "turn": _parse_action_tokens(parts[6]),
        "river": _parse_action_tokens(parts[7]),
        "bankroll": bankroll,
        "winnings": winnings,
        "hole_cards": hole,
    }


def _player_uid(name: str, month: str) -> str:
    # deterministic pseudonymization with source month.
    digest = hashlib.sha256(f"{month}:{name}".encode("utf-8")).hexdigest()
    return f"p_{digest}"


def _token_to_action_type(tok: str, preflop_blinds_seen: List[str]) -> Optional[str]:
    c = tok[:1]
    if c == "B":
        if len(preflop_blinds_seen) == 0:
            preflop_blinds_seen.append("sb")
            return "small_blind"
        if len(preflop_blinds_seen) == 1:
            preflop_blinds_seen.append("bb")
            return "big_blind"
        return "bet"
    if c in ("r", "R"):
        return "raise"
    if c in ("A",):
        return "all_in"
    if c in ("c", "C"):
        return "call"
    if c in ("k", "K"):
        return "check"
    if c in ("f", "q", "Q"):
        return "fold"
    return None


def _street_board(street: str, board_cards: List[str]) -> List[str]:
    if street == "preflop":
        return []
    if street == "flop":
        return board_cards[:3]
    if street == "turn":
        return board_cards[:4]
    if street == "river":
        return board_cards[:5]
    return []


def _build_hand_from_group(hdb: HdbRecord, pdb_rows: List[dict]) -> dict:
    seats_sorted = sorted({int(r["seat"]) for r in pdb_rows if int(r["seat"]) > 0})
    max_seat = max(seats_sorted) if seats_sorted else max(2, int(hdb.num_players))
    max_seats = max(max_seat, int(hdb.num_players), 2)

    players = []
    for r in sorted(pdb_rows, key=lambda x: int(x["seat"])):
        hole = r["hole_cards"] if r["hole_cards"] else None
        players.append(
            {
                "player_uid": _player_uid(str(r["player"]), hdb.month),
                "seat": int(r["seat"]),
                "starting_stack": float(max(0.0, r["bankroll"])),
                "hole_cards": hole,
                "showed_hand": bool(hole),
            }
        )

    streets = []
    for street in ("flop", "turn", "river"):
        b = _street_board(street, hdb.board_cards)
        if b:
            streets.append({"street": street, "board_cards": b})

    actions = []
    act_id = 1
    pot = 0.0
    blind_seen: List[str] = []
    by_street = ("preflop", "flop", "turn", "river")
    for st in by_street:
        for r in sorted(pdb_rows, key=lambda x: int(x["seat"])):
            for tok in r.get(st, []):
                at = _token_to_action_type(tok, blind_seen if st == "preflop" else [])
                if at is None:
                    continue
                amount = 0.0
                if len(tok) > 1 and tok[1:].isdigit():
                    amount = float(tok[1:])
                pot_before = pot
                if at in {"small_blind", "big_blind", "call", "bet", "raise", "all_in"}:
                    pot += max(0.0, amount)
                action = {
                    "action_id": str(act_id),
                    "street": st,
                    "actor_seat": int(r["seat"]),
                    "action_type": at,
                    "amount": round(max(0.0, amount), 4),
                    "raise_to": None,
                    "call_to": None,
                    "normalized_amount_bb": 0.0,
                    "pot_before": round(pot_before, 4),
                    "pot_after": round(pot, 4),
                }
                actions.append(action)
                act_id += 1

    if hdb.street_pots:
        final_est = max(hdb.street_pots.values())
        if final_est > pot and actions:
            actions[-1]["pot_after"] = round(final_est, 4)

    metadata = {
        "game_type": "Hold'em",
        "limit_type": "Unknown",
        "max_seats": int(max_seats),
        "hero_seat": 0,
        "hand_ended_on_street": "",
        "button_seat": 0,
        "sb": 0.01,
        "bb": 0.02,
        "ante": 0.0,
        "rng_seed_commitment": None,
    }

    return {
        "metadata": metadata,
        "players": players,
        "streets": streets,
        "actions": actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": round(max(hdb.street_pots.values()) if hdb.street_pots else 0.0, 4),
            "rake": 0.0,
            "result_reason": "",
            "showdown": any(p.get("showed_hand") for p in players),
        },
        "_provenance": {
            "source": "irc",
            "source_month": hdb.month,
            "source_table_id": hdb.table_id,
            "source_hand_id": hdb.hand_id,
            "parser_version": SCRIPT_VERSION,
            "schema_version": SCHEMA_VERSION,
        },
    }


def _qc_reasons(hand: dict) -> List[str]:
    reasons: List[str] = []
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    if len(players) < 2:
        reasons.append("missing_min_fields_players_lt_2")
    if len(actions) == 0:
        reasons.append("missing_min_fields_no_actions")

    # malformed action tokens / actor seats
    for a in actions:
        if not isinstance(a.get("actor_seat"), int) or a.get("actor_seat", 0) <= 0:
            reasons.append("malformed_action_actor_seat")
            break
        if not isinstance(a.get("street"), str):
            reasons.append("malformed_action_street")
            break

    # invalid card states
    seen_cards = set()
    board_dupe = False
    for st in streets:
        for c in st.get("board_cards", []):
            if _norm_card(str(c)) is None:
                reasons.append("invalid_card_state_bad_board_card")
                board_dupe = True
                break
            if c in seen_cards:
                reasons.append("invalid_card_state_duplicate_board_card")
                board_dupe = True
                break
            seen_cards.add(c)
        if board_dupe:
            break

    # impossible pot evolution
    for a in actions:
        pb = float(a.get("pot_before", 0.0) or 0.0)
        pa = float(a.get("pot_after", 0.0) or 0.0)
        if pa + 1e-9 < pb:
            reasons.append("impossible_pot_evolution")
            break

    # street order monotonic (preflop->flop->turn->river)
    order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    last = -1
    for a in actions:
        v = order.get(str(a.get("street", "")).lower(), None)
        if v is None:
            reasons.append("street_order_unknown_street")
            break
        if v < last:
            reasons.append("street_order_violation")
            break
        last = v

    # sanitizer / aggregator compatibility gate
    try:
        sanitized = sanitize_hand_for_miner(hand)
        row = aggregate_chunk_from_hands([hand])
        if not sanitized or not row:
            reasons.append("missing_min_fields_needed_by_sanitizer_aggregator")
    except Exception:
        reasons.append("missing_min_fields_needed_by_sanitizer_aggregator")

    return sorted(set(reasons))


def _month_from_filename(name: str) -> str:
    # holdem.YYYYMM.tgz
    m = re.search(r"holdem\.(\d{6})\.tgz$", name)
    return m.group(1) if m else "unknown"


def _month_archives(extract_dir: Path) -> List[Path]:
    candidates = sorted(extract_dir.rglob("holdem.*.tgz"))
    return [p for p in candidates if p.is_file()]


def _init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        """
        CREATE TABLE hdb (
            hand_id TEXT PRIMARY KEY,
            table_id TEXT,
            month TEXT,
            num_players INTEGER,
            board_cards_json TEXT,
            street_pots_json TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE pdb (
            hand_id TEXT,
            player TEXT,
            seat INTEGER,
            preflop TEXT,
            flop TEXT,
            turn TEXT,
            river TEXT,
            bankroll REAL,
            winnings REAL,
            hole1 TEXT,
            hole2 TEXT
        );
        """
    )
    conn.execute("CREATE INDEX idx_pdb_hand ON pdb(hand_id);")
    return conn


def _ingest_month_to_db(month_tgz: Path, conn: sqlite3.Connection, log_every: int = 200000) -> dict:
    month = _month_from_filename(month_tgz.name)
    stats = {
        "hdb_lines": 0,
        "hdb_kept": 0,
        "pdb_lines": 0,
        "pdb_kept": 0,
        "month": month,
        "skipped": False,
        "skip_reason": "",
    }
    _log(f"[month:{month}] ingest {month_tgz}")

    try:
        tf = tarfile.open(month_tgz, "r:gz", errorlevel=0)
    except (tarfile.TarError, EOFError, OSError) as e:
        stats["skipped"] = True
        stats["skip_reason"] = f"tar_open_failed: {e}"
        _log(f"[month:{month}] WARN skip month (cannot open archive): {e}")
        return stats

    with tf:
        try:
            members = [m for m in tf.getmembers() if m.isfile()]
        except (tarfile.TarError, EOFError, OSError) as e:
            stats["skipped"] = True
            stats["skip_reason"] = f"tar_read_members_failed: {e}"
            _log(f"[month:{month}] WARN skip month (corrupt/truncated archive): {e}")
            return stats
        hdb_members = [m for m in members if m.name.endswith("/hdb") or m.name.endswith("hdb")]
        pdb_members = [m for m in members if "/pdb/" in m.name]
        if not hdb_members:
            _log(f"[month:{month}] no hdb file found, skip")
            return stats

        hdb_batch = []
        for line in _iter_lines_from_member(tf, hdb_members[0]):
            stats["hdb_lines"] += 1
            rec = _parse_hdb_line(line, month)
            if rec is None:
                continue
            stats["hdb_kept"] += 1
            hdb_batch.append(
                (
                    rec.hand_id,
                    rec.table_id,
                    rec.month,
                    int(rec.num_players),
                    json.dumps(rec.board_cards, ensure_ascii=True),
                    json.dumps(rec.street_pots, ensure_ascii=True),
                )
            )
            if len(hdb_batch) >= 5000:
                conn.executemany(
                    "INSERT OR REPLACE INTO hdb(hand_id, table_id, month, num_players, board_cards_json, street_pots_json) VALUES (?,?,?,?,?,?)",
                    hdb_batch,
                )
                conn.commit()
                hdb_batch.clear()
            if stats["hdb_lines"] % log_every == 0:
                _log(f"[month:{month}] hdb lines={stats['hdb_lines']:,} kept={stats['hdb_kept']:,}")
        if hdb_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO hdb(hand_id, table_id, month, num_players, board_cards_json, street_pots_json) VALUES (?,?,?,?,?,?)",
                hdb_batch,
            )
            conn.commit()

        pdb_batch = []
        for member in pdb_members:
            for line in _iter_lines_from_member(tf, member):
                stats["pdb_lines"] += 1
                rec = _parse_pdb_line(line)
                if rec is None:
                    continue
                stats["pdb_kept"] += 1
                h1, h2 = (rec["hole_cards"] + [None, None])[:2]
                pdb_batch.append(
                    (
                        rec["hand_id"],
                        rec["player"],
                        int(rec["seat"]),
                        json.dumps(rec["preflop"], ensure_ascii=True),
                        json.dumps(rec["flop"], ensure_ascii=True),
                        json.dumps(rec["turn"], ensure_ascii=True),
                        json.dumps(rec["river"], ensure_ascii=True),
                        float(rec["bankroll"]),
                        float(rec["winnings"]),
                        h1,
                        h2,
                    )
                )
                if len(pdb_batch) >= 5000:
                    conn.executemany(
                        "INSERT INTO pdb(hand_id, player, seat, preflop, flop, turn, river, bankroll, winnings, hole1, hole2) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        pdb_batch,
                    )
                    conn.commit()
                    pdb_batch.clear()
                if stats["pdb_lines"] % log_every == 0:
                    _log(f"[month:{month}] pdb lines={stats['pdb_lines']:,} kept={stats['pdb_kept']:,}")
        if pdb_batch:
            conn.executemany(
                "INSERT INTO pdb(hand_id, player, seat, preflop, flop, turn, river, bankroll, winnings, hole1, hole2) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                pdb_batch,
            )
            conn.commit()
    _log(
        f"[month:{month}] ingest done | hdb {stats['hdb_kept']:,}/{stats['hdb_lines']:,} "
        f"| pdb {stats['pdb_kept']:,}/{stats['pdb_lines']:,}"
    )
    return stats


def _stream_emit_from_db(
    conn: sqlite3.Connection,
    accepted_f,
    rejects_f,
    qc_counter: Counter,
    sample_limit: Optional[int],
    log_every_hands: int,
) -> dict:
    q = """
    SELECT
      p.hand_id,
      h.table_id, h.month, h.num_players, h.board_cards_json, h.street_pots_json,
      p.player, p.seat, p.preflop, p.flop, p.turn, p.river, p.bankroll, p.winnings, p.hole1, p.hole2
    FROM pdb p
    JOIN hdb h ON h.hand_id = p.hand_id
    ORDER BY p.hand_id, p.seat
    """
    cur = conn.execute(q)

    produced = 0
    accepted = 0
    rejected = 0
    curr_hand_id = None
    curr_hdb = None
    curr_rows: List[dict] = []

    def flush_one() -> None:
        nonlocal produced, accepted, rejected, curr_hand_id, curr_hdb, curr_rows
        if curr_hand_id is None or curr_hdb is None:
            return
        produced += 1
        hand = _build_hand_from_group(curr_hdb, curr_rows)
        reasons = _qc_reasons(hand)
        if reasons:
            rejected += 1
            for r in reasons:
                qc_counter[r] += 1
            rej = {
                "source_hand_id": curr_hdb.hand_id,
                "source_month": curr_hdb.month,
                "source_table_id": curr_hdb.table_id,
                "reject_reasons": reasons,
            }
            rejects_f.write(json.dumps(rej, ensure_ascii=True) + "\n")
        else:
            accepted += 1
            accepted_f.write(json.dumps(hand, ensure_ascii=True) + "\n")
        if produced % log_every_hands == 0:
            _log(
                f"[emit] hands={produced:,} accepted={accepted:,} rejected={rejected:,} "
                f"accept_rate={(accepted/max(1,produced))*100:.2f}%"
            )

    for row in cur:
        if sample_limit is not None and accepted >= sample_limit:
            break
        (
            hand_id,
            table_id,
            month,
            num_players,
            board_cards_json,
            street_pots_json,
            player,
            seat,
            pre,
            flop,
            turn,
            river,
            bankroll,
            winnings,
            hole1,
            hole2,
        ) = row
        if curr_hand_id is None:
            curr_hand_id = hand_id
            curr_hdb = HdbRecord(
                hand_id=str(hand_id),
                table_id=str(table_id),
                month=str(month),
                num_players=int(num_players),
                board_cards=list(json.loads(board_cards_json or "[]")),
                street_pots=dict(json.loads(street_pots_json or "{}")),
            )
        if hand_id != curr_hand_id:
            flush_one()
            curr_hand_id = hand_id
            curr_hdb = HdbRecord(
                hand_id=str(hand_id),
                table_id=str(table_id),
                month=str(month),
                num_players=int(num_players),
                board_cards=list(json.loads(board_cards_json or "[]")),
                street_pots=dict(json.loads(street_pots_json or "{}")),
            )
            curr_rows = []
        hole_cards = []
        if hole1 and hole2:
            c1 = _norm_card(str(hole1))
            c2 = _norm_card(str(hole2))
            if c1 and c2:
                hole_cards = [c1, c2]
        curr_rows.append(
            {
                "player": str(player),
                "seat": int(seat),
                "preflop": list(json.loads(pre or "[]")),
                "flop": list(json.loads(flop or "[]")),
                "turn": list(json.loads(turn or "[]")),
                "river": list(json.loads(river or "[]")),
                "bankroll": float(bankroll),
                "winnings": float(winnings),
                "hole_cards": hole_cards,
            }
        )
    flush_one()
    _log(
        f"[emit] done | hands={produced:,} accepted={accepted:,} rejected={rejected:,} "
        f"accept_rate={(accepted/max(1,produced))*100:.2f}%"
    )
    return {"hands_total": produced, "accepted": accepted, "rejected": rejected}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-build normalized Poker44 hands from IRC dataset.")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--sample", type=int, default=None, help="Stop after N accepted hands.")
    ap.add_argument("--log-every-lines", type=int, default=200000, help="Line-level ingest log interval.")
    ap.add_argument("--log-every-hands", type=int, default=10000, help="Hand-level emit log interval.")
    ap.add_argument("--archive-path", type=Path, default=ARCHIVE_PATH)
    ap.add_argument("--extract-dir", type=Path, default=EXTRACT_DIR)
    ap.add_argument("--out-accepted", type=Path, default=OUT_ACCEPTED)
    ap.add_argument("--out-rejects", type=Path, default=OUT_REJECTS)
    ap.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    args = ap.parse_args()

    args.out_accepted.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        _download(IRC_URL, args.archive_path)
    if not args.skip_extract:
        _safe_extract_tar_gz(args.archive_path, args.extract_dir)

    months = _month_archives(args.extract_dir)
    if not months:
        raise SystemExit(f"No holdem.*.tgz found under {args.extract_dir}")
    _log(f"[main] month archives: {len(months)}")

    qc_counter: Counter = Counter()
    month_stats = []
    global_counts = Counter()

    with args.out_accepted.open("w", encoding="utf-8") as acc_f, args.out_rejects.open(
        "w", encoding="utf-8"
    ) as rej_f:
        for i, month_tgz in enumerate(months, start=1):
            if args.sample is not None and global_counts["accepted"] >= args.sample:
                _log("[main] sample limit reached; stopping early.")
                break
            _log(f"[main] ({i}/{len(months)}) processing {month_tgz.name}")
            db_path = TMP_DIR / f"{month_tgz.stem}.sqlite"
            conn = _init_db(db_path)
            try:
                ms = _ingest_month_to_db(month_tgz, conn, log_every=args.log_every_lines)
                if ms.get("skipped"):
                    month_stats.append(ms)
                    _log(f"[main] month skipped {month_tgz.name} | reason={ms.get('skip_reason', '')}")
                    continue
                if ms.get("hdb_kept", 0) == 0 or ms.get("pdb_kept", 0) == 0:
                    ms.update({"hands_total": 0, "accepted": 0, "rejected": 0})
                    month_stats.append(ms)
                    _log(f"[main] month skipped {month_tgz.name} | empty hdb/pdb after ingest")
                    continue
                out = _stream_emit_from_db(
                    conn,
                    acc_f,
                    rej_f,
                    qc_counter,
                    sample_limit=(args.sample - global_counts["accepted"]) if args.sample else None,
                    log_every_hands=args.log_every_hands,
                )
                ms.update(out)
                month_stats.append(ms)
                global_counts["hands_total"] += out["hands_total"]
                global_counts["accepted"] += out["accepted"]
                global_counts["rejected"] += out["rejected"]
                _log(
                    f"[main] month done {month_tgz.name} | accepted={out['accepted']:,} "
                    f"rejected={out['rejected']:,}"
                )
            finally:
                conn.close()
                try:
                    db_path.unlink()
                except OSError:
                    pass

    summary = {
        "parser_version": SCRIPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": "IRC",
        "archive": str(args.archive_path),
        "extract_dir": str(args.extract_dir),
        "out_accepted": str(args.out_accepted),
        "out_rejects": str(args.out_rejects),
        "sample_limit": args.sample,
        "global_counts": dict(global_counts),
        "reject_reason_counts": dict(qc_counter),
        "month_stats": month_stats,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    _log(f"[main] summary written: {args.out_summary}")
    _log(
        f"[main] COMPLETE | accepted={global_counts['accepted']:,} "
        f"rejected={global_counts['rejected']:,} total={global_counts['hands_total']:,}"
    )


if __name__ == "__main__":
    main()
