"""Deduplicate unlabeled validator chunks across all log files.

Writes:
    workspace/hybrid/bot_system/data/unlabeled_unique.jsonl
        One line per unique chunk_hash with fields:
            chunk_hash, chunk, risk_score_mean, risk_score_std, n_observed
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "workspace" / "dataset" / "real_distribution"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "unlabeled_unique.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        print(f"[error] no .jsonl files in {args.input_dir}")
        return 1

    by_hash: dict[str, dict] = {}
    scores_by_hash: dict[str, list[float]] = defaultdict(list)
    bad = 0
    total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                h = obj.get("chunk_hash")
                ch = obj.get("chunk")
                rs = obj.get("risk_score")
                if not h or not isinstance(ch, list):
                    bad += 1
                    continue
                if h not in by_hash:
                    by_hash[h] = {"chunk_hash": h, "chunk": ch}
                if isinstance(rs, (int, float)):
                    scores_by_hash[h].append(float(rs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for h, entry in by_hash.items():
            scores = scores_by_hash.get(h, [])
            entry["risk_score_mean"] = statistics.fmean(scores) if scores else None
            entry["risk_score_std"] = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            entry["n_observed"] = len(scores)
            out.write(json.dumps(entry, ensure_ascii=True) + "\n")

    print(f"[done] files={len(files)} lines={total} bad={bad} unique_chunks={len(by_hash)}")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
