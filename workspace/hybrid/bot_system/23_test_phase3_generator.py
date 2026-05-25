"""Phase 3 A/B: Phase-2 passive vs Phase-3 (+ micro-raises, pot build).

Usage:
  python workspace/hybrid/bot_system/23_test_phase3_generator.py
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
from hands_generator.bot_hands.sandbox_poker_bot import BotProfile  # noqa: E402
from may8_validate import validate_vs_may8_hard  # noqa: E402

GOLD = REPO / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
FP = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "may8_hard_target_fingerprint.json"
CHUNK_SIZE = 30
N_CHUNKS = 24

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


def _phase2_only_profile() -> BotProfile:
    p = bot_profile_from_candidate(CAND["profile"], name="probe_p2", passive_may8=True)
    return BotProfile(
        name=p.name,
        passive_may8_mode=True,
        passive_check_bias=p.passive_check_bias,
        passive_bet_scale=p.passive_bet_scale,
        passive_raise_rate=0.0,
        passive_pot_build_mult=1.0,
        tightness=p.tightness,
        aggression=p.aggression,
        bluff_freq=p.bluff_freq,
        max_risk_fraction_of_stack=p.max_risk_fraction_of_stack,
        tilt_factor=p.tilt_factor,
        bet_pot_fraction_small=p.bet_pot_fraction_small,
        bet_pot_fraction_medium=p.bet_pot_fraction_medium,
        bet_pot_fraction_large=p.bet_pot_fraction_large,
        preflop_defend_bias=p.preflop_defend_bias,
        postflop_continue_bias=p.postflop_continue_bias,
        trap_frequency=p.trap_frequency,
    )


def gen_probe(*, phase3: bool) -> pd.DataFrame:
    prof = (
        bot_profile_from_candidate(CAND["profile"], name="probe_p3", passive_may8=True)
        if phase3
        else _phase2_only_profile()
    )
    gen = make_may8_generator(CAND, seed=101)
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


def _print_focus(label: str, scores: dict) -> None:
    print(f"\n  [{label}] raise / pot features:")
    for r in scores.get("features") or []:
        if any(k in r["feature"] for k in ("raise_ratio", "mean_pot_after", "bet_ratio")):
            print(
                f"    {r['feature']:<28} ks={r['ks_gen_vs_hard']:.3f}  "
                f"gen={r['gen_mean']:.3f}  gold={r['hard_mean']:.3f}"
            )


def main() -> int:
    gold = pd.read_parquet(GOLD)
    fp = json.loads(FP.read_text(encoding="utf-8"))

    print("Generating Phase-2 probe (passive, no raises/pot-build) …")
    p2 = gen_probe(phase3=False)
    print("Generating Phase-3 probe (micro-raises + pot build) …")
    p3 = gen_probe(phase3=True)

    s2 = validate_vs_may8_hard(p2, gold, fp)
    s3 = validate_vs_may8_hard(p3, gold, fp)

    print("\n=== Phase 3 A/B (median KS vs May-8 gold bots, lower=better) ===")
    print(f"  phase2:  median_ks={s2['median_ks']}  mean_ks={s2['mean_ks']}  n={s2['n_generated']}")
    print(f"  phase3:  median_ks={s3['median_ks']}  mean_ks={s3['mean_ks']}  n={s3['n_generated']}")

    _print_focus("phase2", s2)
    _print_focus("phase3", s3)

    out = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "phase3_ab_results.json"
    out.write_text(json.dumps({"phase2": s2, "phase3": s3}, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
