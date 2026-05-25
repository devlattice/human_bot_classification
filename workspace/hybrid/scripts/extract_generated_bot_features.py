"""Generate diverse bot hands and extract chunk-level features (parallelized).

Uses multiprocessing across all CPU cores for fast generation.

Usage:
    python workspace/hybrid/scripts/extract_generated_bot_features.py

Output: workspace/hybrid/generated_bot_features.parquet
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
from hands_generator.data_generator import generate_bot_chunk, _default_bot_profiles
from hands_generator.bot_hands.generate_poker_data import BotProfile, _mutate_profile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunk_pipeline import aggregate_chunk_from_raw_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "generated_bot_features.parquet"

CHUNK_SIZE = 60
CHUNKS_PER_PROFILE = 20
N_MUTATIONS_PER_BASE = 59  # 5 base × (1 + 59 mutations) = 300 profiles
SEED = 42
N_WORKERS = min(cpu_count(), 24)


def create_diverse_profiles(seed: int) -> list[BotProfile]:
    """Create 100 diverse bot profiles from 5 base × 20 variants."""
    rng = random.Random(seed)
    base_profiles = _default_bot_profiles()
    all_profiles = []

    for base in base_profiles:
        all_profiles.append(base)
        for i in range(N_MUTATIONS_PER_BASE):
            mutated = _mutate_profile(base, rng)
            mutated.name = f"{base.name}_mut{i}"
            all_profiles.append(mutated)

    return all_profiles


def generate_profile_chunks(args: tuple) -> list[dict]:
    """Worker function: generate all chunks for one profile."""
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
            features["source"] = "generated"
            features["date"] = profile_name
            rows.append(features)
        except Exception:
            continue

    return rows


def main():
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE} hands")
    print(f"Chunks per profile: {CHUNKS_PER_PROFILE}")
    print(f"Workers: {N_WORKERS}")
    print()

    profiles = create_diverse_profiles(SEED)
    print(f"Total profiles: {len(profiles)}")
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
        tasks.append((profile_dict, profile.name, CHUNKS_PER_PROFILE, CHUNK_SIZE, rng.randint(0, 10**9)))

    t0 = time.time()
    all_rows = []

    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_profile_chunks, tasks)):
            all_rows.extend(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] profiles done | {len(all_rows)} chunks | {elapsed:.1f}s")

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Total bot chunks: {len(df)}")
    print(f"Unique profiles: {df['date'].nunique()}")
    print(f"Columns: {len(df.columns)}")
    print(f"other_ratio_mean: {df['other_ratio_mean'].mean():.6f} (should be 0)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
