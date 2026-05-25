"""Phase 2 A/B: Phase-1 engine vs Phase-1 + passive May-8 policy.

Usage:
  python workspace/hybrid/bot_system/22_test_phase2_generator.py
"""

from __future__ import annotations

import contextlib
import io
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
from generator_may8 import bot_profile_from_candidate, make_may8_generator  # noqa: E402
from may8_validate import validate_vs_may8_hard  # noqa: E402

GOLD = REPO / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
FP = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_hard_target_fingerprint.json"
CHUNK_SIZE = 30
N_CHUNKS = 20

CAND = {
    "id": 0,
    "sb": 0.01,
    "bb": 0.02,
    "max_seats": 6,
    "target_players": 6,
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


def gen_probe(*, passive_policy: bool) -> pd.DataFrame:
    prof = bot_profile_from_candidate(
        CAND["profile"], name="probe", passive_may8=passive_policy,
    )
    gen = make_may8_generator(CAND, seed=99)
    total = CHUNK_SIZE * N_CHUNKS
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
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


def _print_check_features(label: str, scores: dict) -> None:
    print(f"\n  [{label}] check / aggression features:")
    for r in scores.get("features") or []:
        if any(k in r["feature"] for k in ("check_ratio", "aggression_factor", "fold_ratio")):
            print(
                f"    {r['feature']:<28} ks={r['ks_gen_vs_hard']:.3f}  "
                f"gen={r['gen_mean']:.3f}  gold={r['hard_mean']:.3f}"
            )


def main() -> int:
    gold = pd.read_parquet(GOLD)
    fp = json.loads(FP.read_text(encoding="utf-8"))

    print("Generating Phase-1 probe (knobs only, no passive policy) …")
    p1 = gen_probe(passive_policy=False)
    print("Generating Phase-2 probe (Phase-1 + passive May-8 policy) …")
    p2 = gen_probe(passive_policy=True)

    s_p1 = validate_vs_may8_hard(p1, gold, fp)
    s_p2 = validate_vs_may8_hard(p2, gold, fp)

    print("\n=== Phase 2 A/B (median KS vs May-8 gold bots, lower=better) ===")
    print(f"  phase1:  median_ks={s_p1['median_ks']}  mean_ks={s_p1['mean_ks']}  n_chunks={s_p1['n_generated']}")
    print(f"  phase2:  median_ks={s_p2['median_ks']}  mean_ks={s_p2['mean_ks']}  n_chunks={s_p2['n_generated']}")

    _print_check_features("phase1", s_p1)
    _print_check_features("phase2", s_p2)

    out = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "phase2_ab_results.json"
    out.write_text(json.dumps({"phase1": s_p1, "phase2": s_p2}, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
