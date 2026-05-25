"""KS sanity: generated may8_matched_bot vs May-8 hard gold bots.

Usage:
  python workspace/hybrid/bot_system/19_validate_may8_generated.py \\
    --generated workspace/hybrid/bot_system/data/may8_reflect_bot_features.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "bot_system"))

from may8_validate import validate_vs_may8_hard  # noqa: E402

DEFAULT_GEN = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_reflect_bot_features.parquet"
DEFAULT_GOLD = REPO / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
DEFAULT_FP = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_hard_target_fingerprint.json"
META = {"label", "date", "chunk_idx", "source"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, default=DEFAULT_GEN)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--fingerprint", type=Path, default=DEFAULT_FP)
    p.add_argument("--out", type=Path, default=REPO / "workspace/hybrid/bot_system/data/may8_generated_validation.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.generated.is_file():
        print(f"[error] missing {args.generated}")
        return 1
    gen = pd.read_parquet(args.generated)
    gold = pd.read_parquet(args.gold)
    fp = json.loads(args.fingerprint.read_text(encoding="utf-8")) if args.fingerprint.is_file() else {}
    summary = validate_vs_may8_hard(gen, gold, fp)
    rows = summary.get("features") or []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[validate] gen={len(gen)}  may8_bot_gold={summary.get('n_may8_bot_gold')}  "
        f"median_ks={summary['median_ks']}"
    )
    print("  best match (low KS):")
    for r in rows[:8]:
        print(f"    {r['feature']:<32} ks={r['ks_gen_vs_hard']:.4f}  gen={r['gen_mean']:.4f}  hard={r['hard_mean']:.4f}")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
