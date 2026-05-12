"""Generate bot hands covering the FULL parameter space via Latin Hypercube Sampling.

Unlike the previous generator (small mutations around 5 base profiles), this script
systematically samples across all 11 BotProfile dimensions, including extreme corners
like passive/tight bots (May-8 style) that were previously missing.

Usage:
    python workspace/hybrid/scripts/generate_full_spectrum_bots.py [--n-profiles 1000] [--chunks-per-profile 10]

Output: workspace/hybrid/full_spectrum_bot_features.parquet
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import asdict

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from hands_generator.data_generator import generate_bot_chunk
from hands_generator.bot_hands.generate_poker_data import BotProfile
from poker44.validator.chunk_features import aggregate_chunk_from_hands

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "workspace" / "hybrid" / "full_spectrum_bot_features.parquet"
PROFILE_LOG_PATH = REPO_ROOT / "workspace" / "hybrid" / "full_spectrum_profiles.json"

PARAM_RANGES = {
    "tightness":                  (0.10, 0.90),
    "aggression":                 (0.05, 0.95),
    "bluff_freq":                 (0.00, 0.25),
    "max_risk_fraction_of_stack": (0.05, 0.40),
    "tilt_factor":                (0.00, 0.40),
    "bet_pot_fraction_small":     (0.10, 0.70),
    "bet_pot_fraction_medium":    (0.20, 1.10),
    "bet_pot_fraction_large":     (0.30, 1.50),
    "preflop_defend_bias":        (-1.00, 1.00),
    "postflop_continue_bias":     (-1.00, 1.00),
    "trap_frequency":             (-1.00, 1.00),
}

ARCHETYPES = {
    "passive_tight":   {"tightness": 0.80, "aggression": 0.15, "bluff_freq": 0.01,
                        "preflop_defend_bias": -0.6, "postflop_continue_bias": -0.4},
    "passive_loose":   {"tightness": 0.25, "aggression": 0.12, "bluff_freq": 0.02,
                        "preflop_defend_bias": 0.5, "postflop_continue_bias": -0.3},
    "aggressive_tight":{"tightness": 0.75, "aggression": 0.90, "bluff_freq": 0.12,
                        "preflop_defend_bias": -0.3, "postflop_continue_bias": 0.3},
    "aggressive_loose":{"tightness": 0.20, "aggression": 0.88, "bluff_freq": 0.18,
                        "preflop_defend_bias": 0.6, "postflop_continue_bias": 0.5},
    "trapper":         {"tightness": 0.55, "aggression": 0.40, "bluff_freq": 0.03,
                        "trap_frequency": 0.8, "postflop_continue_bias": 0.4},
    "maniac":          {"tightness": 0.15, "aggression": 0.92, "bluff_freq": 0.22,
                        "max_risk_fraction_of_stack": 0.35, "tilt_factor": 0.30},
    "nit":             {"tightness": 0.88, "aggression": 0.30, "bluff_freq": 0.00,
                        "max_risk_fraction_of_stack": 0.08, "preflop_defend_bias": -0.8},
    "calling_station": {"tightness": 0.30, "aggression": 0.10, "bluff_freq": 0.01,
                        "preflop_defend_bias": 0.7, "postflop_continue_bias": 0.7,
                        "trap_frequency": -0.5},
}

CHUNK_SIZE = 60
N_WORKERS = min(cpu_count(), 24)


def latin_hypercube_sample(n: int, seed: int) -> np.ndarray:
    """Generate n samples in [0,1]^d via LHS for d = len(PARAM_RANGES)."""
    rng = np.random.RandomState(seed)
    d = len(PARAM_RANGES)
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        for i in range(n):
            result[perm[i], j] = (i + rng.uniform()) / n
    return result


def create_profiles(n_lhs: int, seed: int) -> list[BotProfile]:
    """Create profiles: LHS grid + explicit archetypes + archetype mutations."""
    rng = random.Random(seed)
    param_names = list(PARAM_RANGES.keys())
    param_lo = np.array([PARAM_RANGES[p][0] for p in param_names])
    param_hi = np.array([PARAM_RANGES[p][1] for p in param_names])

    profiles = []

    # 1) LHS profiles
    lhs = latin_hypercube_sample(n_lhs, seed)
    lhs_scaled = param_lo + lhs * (param_hi - param_lo)

    for i in range(n_lhs):
        params = {param_names[j]: float(lhs_scaled[i, j]) for j in range(len(param_names))}
        params["name"] = f"lhs_{i:04d}"
        profiles.append(BotProfile(**params))

    # 2) Archetype cores
    for arch_name, arch_params in ARCHETYPES.items():
        defaults = {p: float((lo + hi) / 2) for p, (lo, hi) in PARAM_RANGES.items()}
        defaults.update(arch_params)
        defaults["name"] = f"arch_{arch_name}"
        profiles.append(BotProfile(**defaults))

    # 3) Archetype mutations (20 variants each)
    for arch_name, arch_params in ARCHETYPES.items():
        for m in range(20):
            defaults = {p: float((lo + hi) / 2) for p, (lo, hi) in PARAM_RANGES.items()}
            defaults.update(arch_params)
            for p in param_names:
                lo, hi = PARAM_RANGES[p]
                noise = rng.uniform(-0.1, 0.1) * (hi - lo)
                defaults[p] = max(lo, min(hi, defaults[p] + noise))
            defaults["name"] = f"arch_{arch_name}_m{m:02d}"
            profiles.append(BotProfile(**defaults))

    return profiles


def generate_profile_chunks(args: tuple) -> list[dict]:
    profile_dict, profile_name, n_chunks, chunk_size, base_seed = args
    profile = BotProfile(**profile_dict)
    rng = random.Random(base_seed)
    rows = []
    for _ in range(n_chunks):
        chunk_seed = rng.randint(0, 10**9)
        try:
            chunk_hands = generate_bot_chunk(chunk_size, [profile], seed=chunk_seed)
            features = aggregate_chunk_from_hands(chunk_hands, skip_sanitize=False)
            features["label"] = 1
            features["source"] = "full_spectrum"
            features["profile"] = profile_name
            rows.append(features)
        except Exception:
            continue
    return rows


def profile_to_dict(p: BotProfile) -> dict:
    return {
        "name": p.name,
        "tightness": p.tightness,
        "aggression": p.aggression,
        "bluff_freq": p.bluff_freq,
        "max_risk_fraction_of_stack": p.max_risk_fraction_of_stack,
        "tilt_factor": p.tilt_factor,
        "bet_pot_fraction_small": p.bet_pot_fraction_small,
        "bet_pot_fraction_medium": p.bet_pot_fraction_medium,
        "bet_pot_fraction_large": p.bet_pot_fraction_large,
        "preflop_defend_bias": p.preflop_defend_bias,
        "postflop_continue_bias": p.postflop_continue_bias,
        "trap_frequency": p.trap_frequency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profiles", type=int, default=1000, help="Number of LHS profiles")
    ap.add_argument("--chunks-per-profile", type=int, default=10, help="Chunks per profile")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Output: {OUTPUT_PATH}")
    print(f"LHS profiles: {args.n_profiles}")
    n_archetypes = len(ARCHETYPES) * (1 + 20)
    total_profiles = args.n_profiles + n_archetypes
    print(f"Archetype profiles: {n_archetypes} ({len(ARCHETYPES)} archetypes × 21)")
    print(f"Total profiles: {total_profiles}")
    print(f"Chunks per profile: {args.chunks_per_profile}")
    print(f"Expected total chunks: {total_profiles * args.chunks_per_profile}")
    print(f"Workers: {N_WORKERS}")
    print()

    profiles = create_profiles(args.n_profiles, args.seed)
    print(f"Created {len(profiles)} profiles")

    profile_log = [profile_to_dict(p) for p in profiles]
    PROFILE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_LOG_PATH.write_text(json.dumps(profile_log, indent=2), encoding="utf-8")
    print(f"Saved profile log: {PROFILE_LOG_PATH}")

    rng = random.Random(args.seed + 1)
    tasks = []
    for p in profiles:
        pd_dict = profile_to_dict(p)
        tasks.append((pd_dict, p.name, args.chunks_per_profile, CHUNK_SIZE, rng.randint(0, 10**9)))

    t0 = time.time()
    all_rows = []

    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_profile_chunks, tasks)):
            all_rows.extend(result)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] {len(all_rows)} chunks | {elapsed:.0f}s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    df = pd.DataFrame(all_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total bot chunks: {len(df)}")
    print(f"Unique profiles: {df['profile'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    agg_range = df[["aggression"]].describe() if "aggression" in df.columns else None
    print(f"\nParameter coverage in generated data:")
    for param in ["tightness", "aggression", "bluff_freq"]:
        if param not in df.columns:
            vals = [profile_to_dict(p)[param] for p in profiles]
            print(f"  {param}: [{min(vals):.3f}, {max(vals):.3f}]")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
