"""Phase 1 A/B: legacy generator vs May-8-conditioned (locked stakes + BB stacks).

Usage:
  python workspace/hybrid/bot_system/21_test_phase1_generator.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "bot_system"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

from chunk_pipeline import aggregate_chunk_from_raw_hands  # noqa: E402
from generator_may8 import make_may8_generator  # noqa: E402
from hands_generator.bot_hands.generate_poker_data import PokerHandGenerator  # noqa: E402
from hands_generator.bot_hands.sandbox_poker_bot import BotProfile  # noqa: E402
from may8_validate import validate_vs_may8_hard  # noqa: E402

GOLD = REPO / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
FP = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_hard_target_fingerprint.json"
CHUNK_SIZE = 30
N_CHUNKS = 20


def gen_probe(legacy: bool) -> pd.DataFrame:
    cand = {
        "id": 0,
        "sb": 0.01,
        "bb": 0.02,
        "max_seats": 6,
        "profile": {
            "tightness": 0.88,
            "aggression": 0.12,
            "bluff_freq": 0.02,
            "max_risk_fraction_of_stack": 0.25,
            "tilt_factor": 0.05,
            "bet_pot_fraction_small": 0.22,
            "bet_pot_fraction_medium": 0.40,
            "bet_pot_fraction_large": 0.55,
            "preflop_defend_bias": -0.5,
            "postflop_continue_bias": -0.4,
            "trap_frequency": 0.05,
        },
    }
    prof = BotProfile(name="probe", **cand["profile"])
    if legacy:
        gen = PokerHandGenerator(sb=0.01, bb=0.02, max_seats=6, seed=99)
    else:
        gen = make_may8_generator(cand, seed=99)

    total = CHUNK_SIZE * N_CHUNKS
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        hands = gen.generate_hands(
            num_hands_to_play=total + CHUNK_SIZE * 2,
            num_hands_to_select=total,
            bot_profiles=[prof],
            output_file=path,
            hands_per_session=CHUNK_SIZE,
        )
    rows = []
    for i in range(0, len(hands), CHUNK_SIZE):
        chunk = hands[i : i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            break
        row = aggregate_chunk_from_raw_hands(chunk)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    gold = pd.read_parquet(GOLD)
    fp = json.loads(FP.read_text(encoding="utf-8"))

    print("Generating legacy probe …")
    leg = gen_probe(legacy=True)
    print("Generating Phase-1 probe …")
    p1 = gen_probe(legacy=False)

    s_leg = validate_vs_may8_hard(leg, gold, fp)
    s_p1 = validate_vs_may8_hard(p1, gold, fp)

    print("\n=== Phase 1 A/B (median KS vs May-8 gold bots, lower=better) ===")
    print(f"  legacy:  median_ks={s_leg['median_ks']}  mean_ks={s_leg['mean_ks']}  n_chunks={s_leg['n_generated']}")
    print(f"  phase1:  median_ks={s_p1['median_ks']}  mean_ks={s_p1['mean_ks']}  n_chunks={s_p1['n_generated']}")

    for label, s in ("legacy", s_leg), ("phase1", s_p1):
        print(f"\n  [{label}] sizing features:")
        for r in s.get("features") or []:
            if any(k in r["feature"] for k in ("mean_norm_bb", "bet_size_mean", "pot_growth")):
                print(
                    f"    {r['feature']:<24} ks={r['ks_gen_vs_hard']:.3f}  "
                    f"gen={r['gen_mean']:.2f}  gold={r['hard_mean']:.2f}"
                )

    out = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "phase1_ab_results.json"
    out.write_text(json.dumps({"legacy": s_leg, "phase1": s_p1}, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
