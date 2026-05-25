"""Generate bot hands calibrated to match gold bot behavioral distribution.

Fixes two issues vs default generator:
1. Profile parameters tuned to gold bot action ratios (less folding, more aggression)
2. Bet sizing fractions increased to match gold bot ~18BB average bet size

Usage:
    python workspace/hybrid/scripts/extract_calibrated_bot_features.py

Output: workspace/hybrid/calibrated_bot_features.parquet
"""

import os
import sys
import time
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from hands_generator.data_generator import generate_bot_chunk
from hands_generator.bot_hands.generate_poker_data import BotProfile, _mutate_profile, _clamp_float
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_pipeline import aggregate_chunk_from_raw_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "calibrated_bot_features.parquet"

CHUNK_SIZE = 60
CHUNKS_PER_PROFILE = 20
SEED = 777
N_WORKERS = min(cpu_count(), 24)


def _gold_calibrated_base_profiles() -> list[BotProfile]:
    """5 base profiles calibrated to gold bot behavioral statistics.

    Gold bot targets:
        fold_ratio ~0.28, aggression_factor ~1.68, action_entropy ~1.79
        bet_size_mean ~18BB, call_ratio ~0.15, raise_ratio ~0.21, bet_ratio ~0.18
    """
    return [
        BotProfile(
            name="gold_balanced",
            tightness=0.38,
            aggression=0.75,
            bluff_freq=0.08,
            preflop_defend_bias=0.20,
            postflop_continue_bias=0.20,
            trap_frequency=0.00,
            bet_pot_fraction_small=0.55,
            bet_pot_fraction_medium=0.85,
            bet_pot_fraction_large=1.10,
        ),
        BotProfile(
            name="gold_aggressive",
            tightness=0.35,
            aggression=0.82,
            bluff_freq=0.10,
            preflop_defend_bias=0.25,
            postflop_continue_bias=0.25,
            trap_frequency=-0.05,
            bet_pot_fraction_small=0.60,
            bet_pot_fraction_medium=0.90,
            bet_pot_fraction_large=1.20,
        ),
        BotProfile(
            name="gold_tricky",
            tightness=0.40,
            aggression=0.70,
            bluff_freq=0.12,
            preflop_defend_bias=0.18,
            postflop_continue_bias=0.15,
            trap_frequency=0.10,
            bet_pot_fraction_small=0.50,
            bet_pot_fraction_medium=0.80,
            bet_pot_fraction_large=1.05,
        ),
        BotProfile(
            name="gold_loose_aggro",
            tightness=0.32,
            aggression=0.85,
            bluff_freq=0.09,
            preflop_defend_bias=0.30,
            postflop_continue_bias=0.28,
            trap_frequency=-0.08,
            bet_pot_fraction_small=0.65,
            bet_pot_fraction_medium=0.95,
            bet_pot_fraction_large=1.25,
        ),
        BotProfile(
            name="gold_solid",
            tightness=0.42,
            aggression=0.72,
            bluff_freq=0.06,
            preflop_defend_bias=0.15,
            postflop_continue_bias=0.18,
            trap_frequency=0.05,
            bet_pot_fraction_small=0.50,
            bet_pot_fraction_medium=0.82,
            bet_pot_fraction_large=1.08,
        ),
    ]


def _calibrated_mutate(profile: BotProfile, rng: random.Random) -> BotProfile:
    """Mutation with tighter bounds to stay near gold-calibrated zone."""
    return BotProfile(
        name=f"{profile.name}_v{rng.randint(1, 9999)}",
        tightness=_clamp_float(profile.tightness + rng.uniform(-0.05, 0.05), 0.28, 0.48),
        aggression=_clamp_float(profile.aggression + rng.uniform(-0.08, 0.08), 0.62, 0.92),
        bluff_freq=_clamp_float(profile.bluff_freq + rng.uniform(-0.03, 0.03), 0.03, 0.15),
        max_risk_fraction_of_stack=_clamp_float(0.22 + rng.uniform(-0.05, 0.05), 0.14, 0.32),
        tilt_factor=0.0,
        bet_pot_fraction_small=_clamp_float(
            profile.bet_pot_fraction_small + rng.uniform(-0.08, 0.08), 0.40, 0.75
        ),
        bet_pot_fraction_medium=_clamp_float(
            profile.bet_pot_fraction_medium + rng.uniform(-0.10, 0.10), 0.70, 1.10
        ),
        bet_pot_fraction_large=_clamp_float(
            profile.bet_pot_fraction_large + rng.uniform(-0.12, 0.12), 0.90, 1.40
        ),
        preflop_defend_bias=_clamp_float(
            profile.preflop_defend_bias + rng.uniform(-0.10, 0.10), 0.05, 0.40
        ),
        postflop_continue_bias=_clamp_float(
            profile.postflop_continue_bias + rng.uniform(-0.10, 0.10), 0.05, 0.35
        ),
        trap_frequency=_clamp_float(
            profile.trap_frequency + rng.uniform(-0.08, 0.08), -0.15, 0.15
        ),
    )


def create_calibrated_profiles(seed: int, n_mutations: int = 59) -> list[BotProfile]:
    """Create 300 gold-calibrated profiles (5 base × 60 variants)."""
    rng = random.Random(seed)
    base_profiles = _gold_calibrated_base_profiles()
    all_profiles = []

    for base in base_profiles:
        all_profiles.append(base)
        for i in range(n_mutations):
            mutated = _calibrated_mutate(base, rng)
            mutated.name = f"{base.name}_mut{i}"
            all_profiles.append(mutated)

    return all_profiles


def generate_profile_chunks(args: tuple) -> list[dict]:
    """Worker: generate chunks for one profile."""
    profile_dict, profile_name, n_chunks, chunk_size, base_seed = args

    profile = BotProfile(**profile_dict)
    rng = random.Random(base_seed)
    rows = []

    for _ in range(n_chunks):
        chunk_seed = rng.randint(0, 10**9)
        try:
            chunk_hands = generate_bot_chunk(chunk_size, [profile], seed=chunk_seed)
            features = aggregate_chunk_from_raw_hands(chunk_hands)
            features["label"] = 1  # bot
            features["source"] = "calibrated"
            features["date"] = profile_name
            rows.append(features)
        except Exception:
            continue

    return rows


def main():
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunks per profile: {CHUNKS_PER_PROFILE}")
    print(f"Workers: {N_WORKERS}")
    print()

    profiles = create_calibrated_profiles(SEED)
    print(f"Total calibrated profiles: {len(profiles)}")
    print(f"Expected chunks: {len(profiles) * CHUNKS_PER_PROFILE}")
    print()

    rng = random.Random(SEED + 1)
    tasks = []
    for profile in profiles:
        profile_dict = {
            "name": profile.name,
            "tightness": profile.tightness,
            "aggression": profile.aggression,
            "bluff_freq": profile.bluff_freq,
            "max_risk_fraction_of_stack": profile.max_risk_fraction_of_stack,
            "tilt_factor": profile.tilt_factor,
            "bet_pot_fraction_small": profile.bet_pot_fraction_small,
            "bet_pot_fraction_medium": profile.bet_pot_fraction_medium,
            "bet_pot_fraction_large": profile.bet_pot_fraction_large,
            "preflop_defend_bias": profile.preflop_defend_bias,
            "postflop_continue_bias": profile.postflop_continue_bias,
            "trap_frequency": profile.trap_frequency,
        }
        tasks.append((
            profile_dict, profile.name, CHUNKS_PER_PROFILE, CHUNK_SIZE,
            rng.randint(0, 10**9)
        ))

    t0 = time.time()
    all_rows = []

    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_profile_chunks, tasks)):
            all_rows.extend(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] profiles | {len(all_rows)} chunks | {elapsed:.1f}s")

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Total bot chunks: {len(df)}")
    print(f"Unique profiles: {df['date'].nunique()}")
    print(f"Columns: {len(df.columns)}")
    print()

    # Quick comparison to gold bot targets
    print("Calibration check vs gold bot targets:")
    print(f"  fold_ratio_mean:   {df['fold_ratio_mean'].mean():.4f}  (gold: 0.2833)")
    print(f"  aggression_factor: {df['aggression_factor_mean'].mean():.4f}  (gold: 1.6825)")
    print(f"  action_entropy:    {df['action_entropy_mean'].mean():.4f}  (gold: 1.7881)")
    print(f"  bet_size_mean:     {df['bet_size_mean_mean'].mean():.4f}  (gold: 18.7304)")
    print(f"  mean_norm_bb:      {df['mean_norm_bb_mean'].mean():.4f}  (gold: 9.4263)")
    print(f"  call_ratio_mean:   {df['call_ratio_mean'].mean():.4f}  (gold: 0.1532)")
    print(f"  raise_ratio_mean:  {df['raise_ratio_mean'].mean():.4f}  (gold: 0.2116)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
