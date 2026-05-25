"""Generate targeted synthetic bot chunks from top-K matched profiles.

- Loads matched_profiles.json (Step 3).
- For each top-K profile, perturbs every knob within ±jitter.
- Runs the generator with the candidate's game-config (max_seats, sb, bb).
- Packs hands into chunks of CHUNK_SIZE and saves features.

Writes:
    workspace/hybrid/bot_system/data/targeted_bot_features.parquet
    (label=1, source="live_matched_bot")
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid"))

from chunk_pipeline import aggregate_chunk_from_raw_hands  # noqa: E402
from generator_may8 import bot_profile_from_candidate, make_may8_generator  # noqa: E402

DEFAULT_MATCHED = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "matched_profiles.json"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "targeted_bot_features.parquet"

CHUNK_SIZE = 30

PROFILE_KNOB_RANGES = {
    "tightness": (0.20, 0.95),
    "aggression": (0.10, 0.95),
    "bluff_freq": (0.00, 0.40),
    "max_risk_fraction_of_stack": (0.05, 0.80),
    "tilt_factor": (0.00, 0.40),
    "bet_pot_fraction_small": (0.10, 0.60),
    "bet_pot_fraction_medium": (0.30, 0.80),
    "bet_pot_fraction_large": (0.55, 1.00),
    "preflop_defend_bias": (-0.70, 0.70),
    "postflop_continue_bias": (-0.70, 0.70),
    "trap_frequency": (0.00, 0.60),
}

PROFILE_KNOB_RANGES_PASSIVE = {
    "tightness": (0.65, 0.99),
    "aggression": (0.02, 0.32),
    "bluff_freq": (0.00, 0.12),
    "max_risk_fraction_of_stack": (0.05, 0.45),
    "tilt_factor": (0.00, 0.18),
    "bet_pot_fraction_small": (0.08, 0.42),
    "bet_pot_fraction_medium": (0.22, 0.58),
    "bet_pot_fraction_large": (0.42, 0.82),
    "preflop_defend_bias": (-0.80, -0.08),
    "postflop_continue_bias": (-0.65, 0.22),
    "trap_frequency": (0.00, 0.28),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _perturb(
    profile: dict, jitter: float, rng: random.Random,
    ranges: dict[str, tuple[float, float]],
) -> dict:
    out = dict(profile)
    for k, (lo, hi) in ranges.items():
        if k not in out:
            continue
        span = hi - lo
        out[k] = _clamp(out[k] + rng.uniform(-jitter, jitter) * span, lo, hi)
    return out


def _gen_chunks(args) -> list[dict]:
    cand, n_chunks, seed, passive, legacy = args
    ranges = PROFILE_KNOB_RANGES_PASSIVE if passive else PROFILE_KNOB_RANGES
    profile_kwargs = cand["profile"]
    p = bot_profile_from_candidate(
        profile_kwargs, name=f"live_match_{cand.get('id', 'x')}", passive_may8=passive,
    )
    if legacy:
        from hands_generator.bot_hands.generate_poker_data import PokerHandGenerator  # noqa: E402

        gen = PokerHandGenerator(
            sb=cand["sb"],
            bb=cand["bb"],
            max_seats=cand["max_seats"],
            rake_rate=0.05,
            seed=seed,
        )
    else:
        gen = make_may8_generator(cand, seed=seed)
    total_hands = CHUNK_SIZE * n_chunks
    # Cap the play budget so extreme profiles can't trap the outer loop.
    play_budget = min(total_hands + CHUNK_SIZE * 4, total_hands * 2)
    tmp = tempfile.NamedTemporaryFile(prefix=f"_gen_{os.getpid()}_", suffix=".json", delete=False)
    tmp.close()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hands = gen.generate_hands(
                num_hands_to_play=play_budget,
                num_hands_to_select=total_hands,
                bot_profiles=[p],
                output_file=tmp.name,
                hands_per_session=CHUNK_SIZE,
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    rows: list[dict] = []
    if not hands:
        return rows
    for i in range(0, len(hands), CHUNK_SIZE):
        chunk = hands[i : i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            break
        try:
            row = aggregate_chunk_from_raw_hands(chunk)
        except Exception:
            continue
        if not row:
            continue
        row["label"] = 1
        if cand.get("_source_tag"):
            row["source"] = cand["_source_tag"]
        else:
            row["source"] = "passive_matched_bot" if passive else "live_matched_bot"
        row["matched_profile_id"] = cand.get("id")
        row["max_seats"] = cand["max_seats"]
        row["sb"] = cand["sb"]
        row["bb"] = cand["bb"]
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-k", type=int, default=15, help="Use top-K matched profiles as seeds")
    p.add_argument("--perturbations-per-seed", type=int, default=8, help="Jittered copies of each seed")
    p.add_argument("--chunks-per-profile", type=int, default=30, help="Chunks per (seed × perturbation)")
    p.add_argument("--jitter", type=float, default=0.10, help="±jitter fraction of each knob range")
    p.add_argument("--workers", type=int, default=min(8, max(1, mp.cpu_count() // 2)))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--per-job-timeout", type=float, default=60.0, help="Max seconds per generation job")
    p.add_argument(
        "--passive",
        action="store_true",
        help="Use passive knob ranges for perturbations + source=passive_matched_bot.",
    )
    p.add_argument(
        "--source-tag",
        type=str,
        default="",
        help="Override parquet source column (e.g. may8_matched_bot).",
    )
    p.add_argument(
        "--legacy-generator",
        action="store_true",
        help="Use pre-Phase-1 generator (resamples stakes; absolute hero stacks).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    matched = json.loads(args.matched.read_text())
    seeds = matched["top"][: args.top_k]
    if not seeds:
        print("[error] no matched profiles available")
        return 1
    rng = random.Random(args.seed)
    ranges = PROFILE_KNOB_RANGES_PASSIVE if args.passive else PROFILE_KNOB_RANGES

    source_tag = (args.source_tag or "").strip()
    legacy = bool(args.legacy_generator)
    jobs: list[tuple[dict, int, int, bool, bool]] = []
    for s in seeds:
        cand = dict(s["candidate"])
        if source_tag:
            cand["_source_tag"] = source_tag
        # Include the un-perturbed seed itself once.
        jobs.append((cand, args.chunks_per_profile, rng.randint(0, 10**9), args.passive, legacy))
        for _ in range(args.perturbations_per_seed):
            cand_j = {
                **cand,
                "profile": _perturb(cand["profile"], args.jitter, rng, ranges),
            }
            jobs.append((cand_j, args.chunks_per_profile, rng.randint(0, 10**9), args.passive, legacy))

    print(f"[plan] seeds={len(seeds)} perturbations={args.perturbations_per_seed} "
          f"chunks/profile={args.chunks_per_profile} total_jobs={len(jobs)}  workers={args.workers}")

    t0 = time.time()
    all_rows: list[dict] = []
    failed = 0
    timed_out = 0
    ctx = mp.get_context("spawn")
    if args.workers <= 1:
        for i, j in enumerate(jobs):
            try:
                r = _gen_chunks(j)
            except Exception as e:
                failed += 1
                print(f"  [job {i+1}] FAILED: {e}")
                continue
            all_rows.extend(r)
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(jobs)}] cumulative_chunks={len(all_rows)}")
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_gen_chunks, j): idx for idx, j in enumerate(jobs)}
            done_count = 0
            for fut in cf.as_completed(futures):
                idx = futures[fut]
                done_count += 1
                try:
                    r = fut.result(timeout=args.per_job_timeout)
                    all_rows.extend(r)
                except cf.TimeoutError:
                    timed_out += 1
                    fut.cancel()
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"  [job {idx}] ERROR: {e}")
                if done_count % 10 == 0 or done_count == 1 or done_count == len(jobs):
                    elapsed = time.time() - t0
                    print(
                        f"  [{done_count}/{len(jobs)}] chunks={len(all_rows)} "
                        f"failed={failed} timeout={timed_out} elapsed={elapsed:.0f}s"
                    )

    elapsed = time.time() - t0
    if not all_rows:
        print("[error] no chunks generated")
        return 1
    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[done] {elapsed:.1f}s  chunks={len(df)}  cols={df.shape[1]}  out={args.out}")
    print("Per-max_seats distribution:")
    print(df.groupby("max_seats").size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
