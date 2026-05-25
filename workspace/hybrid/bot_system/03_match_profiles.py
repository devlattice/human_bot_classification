"""LHS sweep over BotProfile + game config; score against a fingerprint JSON.

For each candidate (BotProfile knobs + max_seats + stake + stack range):
    - Generate a small batch of hands.
    - Pack into "chunks" of CHUNK_SIZE hands each.
    - Aggregate features per chunk.
    - Compute weighted Euclidean distance to the fingerprint
      (using 1 / fingerprint_std as weights).
    - Keep top-K candidates.

Options:
    --passive       Ultra-passive knob ranges (May-8 hard bots).
    --fp-cols robust  Distance only on ROBUST_FEATURES (production RF inputs).
    --stakes micro    Blind pairs 0.01/0.02 and 0.05/0.10 only.

Writes:
    workspace/hybrid/bot_system/data/matched_profiles.json (or --out path)
"""

from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from hands_generator.bot_hands.sandbox_poker_bot import BotProfile  # noqa: E402
from generator_may8 import bot_profile_from_candidate, make_may8_generator  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid"))
from chunk_pipeline import aggregate_chunk_from_raw_hands  # noqa: E402

DEFAULT_FP = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "live_bot_fingerprint.json"
DEFAULT_OUT = REPO_ROOT / "workspace" / "hybrid" / "bot_system" / "data" / "matched_profiles.json"


# ---------- LHS sampling -----------------------------------------------------

PROFILE_KNOB_RANGES: dict[str, tuple[float, float]] = {
    "tightness": (0.30, 0.85),
    "aggression": (0.20, 0.85),
    "bluff_freq": (0.00, 0.30),
    "max_risk_fraction_of_stack": (0.05, 0.60),
    "tilt_factor": (0.00, 0.30),
    "bet_pot_fraction_small": (0.20, 0.50),
    "bet_pot_fraction_medium": (0.40, 0.70),
    "bet_pot_fraction_large": (0.65, 1.00),
    "preflop_defend_bias": (-0.50, 0.50),
    "postflop_continue_bias": (-0.50, 0.50),
    "trap_frequency": (0.00, 0.50),
}

# Ultra-passive / nit regime (May-8 hard bots): low aggression, high tightness.
PROFILE_KNOB_RANGES_PASSIVE: dict[str, tuple[float, float]] = {
    "tightness": (0.70, 0.99),
    "aggression": (0.02, 0.30),
    "bluff_freq": (0.00, 0.10),
    "max_risk_fraction_of_stack": (0.05, 0.40),
    "tilt_factor": (0.00, 0.15),
    "bet_pot_fraction_small": (0.10, 0.38),
    "bet_pot_fraction_medium": (0.25, 0.55),
    "bet_pot_fraction_large": (0.45, 0.78),
    "preflop_defend_bias": (-0.78, -0.12),
    "postflop_continue_bias": (-0.60, 0.18),
    "trap_frequency": (0.00, 0.22),
}

# Live evidence pointed to 9-max, deep stacks; we still include alternates
# so the matcher discovers the actual best, not what we guessed.
GAME_MAX_SEATS = [6, 9]
GAME_STAKES = [(0.01, 0.02), (0.05, 0.10), (0.25, 0.50)]  # (sb, bb)
# Validator / gold-style micro stakes only (no 0.25/0.50) — use with --stakes micro
GAME_STAKES_MICRO = [(0.01, 0.02), (0.05, 0.10)]


def lhs(n_samples: int, n_dims: int, rng: random.Random) -> np.ndarray:
    """Basic Latin Hypercube Sample of shape (n_samples, n_dims) in [0, 1)."""
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = np.array([[rng.random() for _ in range(n_dims)] for _ in range(n_samples)])
    rdpoints = cut[:n_samples, None] + u * (cut[1:, None] - cut[:n_samples, None])
    out = np.empty_like(rdpoints)
    for j in range(n_dims):
        col = rdpoints[:, j].copy()
        rng.shuffle(col)
        out[:, j] = col
    return out


def make_candidates(
    n: int,
    rng: random.Random,
    ranges: dict[str, tuple[float, float]],
    stakes: list[tuple[float, float]] | None = None,
) -> list[dict]:
    knobs = list(ranges.keys())
    grid = lhs(n, len(knobs), rng)
    stake_list = stakes if stakes is not None else GAME_STAKES
    out: list[dict] = []
    for i, row in enumerate(grid):
        profile = {}
        for j, key in enumerate(knobs):
            lo, hi = ranges[key]
            profile[key] = lo + (hi - lo) * float(row[j])
        max_seats = rng.choice(GAME_MAX_SEATS)
        sb, bb = rng.choice(stake_list)
        out.append({
            "id": i,
            "profile": profile,
            "max_seats": max_seats,
            "sb": sb,
            "bb": bb,
        })
    return out


# ---------- Generation + scoring worker -------------------------------------

CHUNK_SIZE = 30  # hands per chunk; close enough for fingerprint matching
NUM_CHUNKS_PER_CANDIDATE = 3  # default; override via --chunks-per-candidate


def _score_candidate(
    cand: dict,
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    feature_cols: list[str],
    seed: int,
    *,
    n_chunks: int = NUM_CHUNKS_PER_CANDIDATE,
    feature_weights: dict[str, float] | None = None,
    passive_may8: bool = False,
) -> dict:
    p = bot_profile_from_candidate(
        cand["profile"], name=f"cand_{cand['id']}", passive_may8=passive_may8,
    )
    gen = make_may8_generator(cand, seed=seed)
    total_hands = CHUNK_SIZE * n_chunks
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f"_disc_{os.getpid()}_{cand['id']}_", suffix=".json", delete=False
    )
    tmp_file.close()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hands = gen.generate_hands(
                num_hands_to_play=total_hands * 2,
                num_hands_to_select=total_hands,
                bot_profiles=[p],
                output_file=tmp_file.name,
                hands_per_session=CHUNK_SIZE,
            )
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass
    if not hands or len(hands) < CHUNK_SIZE:
        return {"id": cand["id"], "distance": float("inf"), "candidate": cand, "n_hands": len(hands or [])}

    dists = []
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
        d2 = 0.0
        wsum = 0.0
        for c in feature_cols:
            if c not in row:
                continue
            mu = feature_means[c]
            sd = max(feature_stds[c], 1e-3)
            v = float(row.get(c, 0.0))
            if not np.isfinite(v):
                continue
            w = float((feature_weights or {}).get(c, 1.0))
            d2 += w * ((v - mu) / sd) ** 2
            wsum += w
        if wsum <= 0:
            continue
        dists.append((d2 / wsum) ** 0.5)

    if not dists:
        return {"id": cand["id"], "distance": float("inf"), "candidate": cand, "n_hands": len(hands)}

    return {
        "id": cand["id"],
        "distance": float(np.mean(dists)),
        "candidate": cand,
        "n_hands": len(hands),
        "n_chunks_scored": len(dists),
    }


def _worker(args):
    cand, fp_means, fp_stds, fp_cols, seed, n_chunks, fp_weights, passive_may8 = args
    try:
        return _score_candidate(
            cand, fp_means, fp_stds, fp_cols, seed,
            n_chunks=n_chunks, feature_weights=fp_weights, passive_may8=passive_may8,
        )
    except Exception as e:
        return {"id": cand["id"], "distance": float("inf"), "candidate": cand, "error": str(e)}


# ---------- Main -------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fp", type=Path, default=DEFAULT_FP)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-candidates", type=int, default=120)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--passive",
        action="store_true",
        help="Use PROFILE_KNOB_RANGES_PASSIVE (ultra-passive bot LHS).",
    )
    p.add_argument(
        "--fp-cols",
        choices=("all", "robust", "may8"),
        default="all",
        help="robust: ROBUST_FEATURES only; may8: fingerprint match_feature_cols (weighted).",
    )
    p.add_argument(
        "--chunks-per-candidate",
        type=int,
        default=NUM_CHUNKS_PER_CANDIDATE,
        help="Chunks simulated per LHS candidate (more = stabler distance).",
    )
    p.add_argument(
        "--refine-top",
        type=int,
        default=0,
        help="If >0, re-score this many best candidates with --refine-chunks each.",
    )
    p.add_argument(
        "--refine-chunks",
        type=int,
        default=10,
        help="Chunks per candidate during refine pass.",
    )
    p.add_argument(
        "--stakes",
        choices=("all", "micro"),
        default="all",
        help="micro: blind pairs 0.01/0.02 and 0.05/0.10 only (no 0.25/0.50).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fp = json.loads(args.fp.read_text())
    fp_means = fp["feature_means"]
    fp_stds = fp["feature_stds"]
    fp_weights: dict[str, float] = dict(fp.get("feature_weights") or {})
    fp_cols = [c for c in fp["feature_cols"] if c in fp_means]

    if args.fp_cols == "may8":
        match_cols = [c for c in fp.get("match_feature_cols") or [] if c in fp_means]
        if not match_cols:
            print("[error] fingerprint missing match_feature_cols; re-run 06 with --ranking")
            return 2
        fp_cols = match_cols
        fp_weights = {c: float(fp_weights.get(c, 1.0)) for c in fp_cols}
        print(f"[fp-cols] may8-ranked: {len(fp_cols)} dims (AUC-weighted distance)")
    elif args.fp_cols == "robust":
        sys.path.insert(0, str(REPO_ROOT / "workspace" / "hybrid" / "scripts"))
        import train_production_model as tpm  # noqa: E402

        robust = set(tpm.ROBUST_FEATURES)
        fp_cols = [c for c in fp_cols if c in robust]
        fp_weights = {c: fp_weights.get(c, 1.0) for c in fp_cols}
        print(f"[fp-cols] robust: {len(fp_cols)} dims (intersection with fingerprint)")
        if not fp_cols:
            print("[error] no overlap between fingerprint and ROBUST_FEATURES")
            return 2

    print(f"[fingerprint] {len(fp_cols)} feature dims  n_bot={fp.get('n_bot')}")
    if fp_weights:
        top_w = sorted(fp_weights.items(), key=lambda x: -x[1])[:5]
        print(f"[weights] top: {', '.join(f'{k}={v:.2f}' for k, v in top_w)}")

    rng = random.Random(args.seed)
    ranges = PROFILE_KNOB_RANGES_PASSIVE if args.passive else PROFILE_KNOB_RANGES
    stake_list = GAME_STAKES_MICRO if args.stakes == "micro" else GAME_STAKES
    candidates = make_candidates(args.n_candidates, rng, ranges, stakes=stake_list)
    mode = "passive" if args.passive else "default"
    print(f"[lhs] mode={mode}  stakes={args.stakes}  {len(candidates)} candidates "
          f"(workers={args.workers})")

    n_chunks = max(1, args.chunks_per_candidate)
    passive_policy = bool(args.passive)
    payloads = [
        (c, fp_means, fp_stds, fp_cols, args.seed + 1000 + c["id"], n_chunks, fp_weights, passive_policy)
        for c in candidates
    ]
    t0 = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for i, p in enumerate(payloads):
            r = _worker(p)
            results.append(r)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(payloads)}] last_dist={r.get('distance', float('inf')):.4f}")
    else:
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, payloads, chunksize=1)):
                results.append(r)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  [{i+1}/{len(payloads)}] last_dist={r.get('distance', float('inf')):.4f}")

    elapsed = time.time() - t0
    results.sort(key=lambda r: r.get("distance", float("inf")))

    if args.refine_top > 0:
        refine_n = min(args.refine_top, len(results))
        print(f"\n[refine] re-scoring top {refine_n} with {args.refine_chunks} chunks each …")
        refined: list[dict] = []
        for r in results[:refine_n]:
            cand = r["candidate"]
            rr = _score_candidate(
                cand, fp_means, fp_stds, fp_cols,
                args.seed + 90000 + cand["id"],
                n_chunks=max(1, args.refine_chunks),
                feature_weights=fp_weights,
                passive_may8=passive_policy,
            )
            refined.append(rr)
            print(f"  id={rr['id']}  coarse={r['distance']:.4f}  refined={rr['distance']:.4f}")
        rest = results[refine_n:]
        results = sorted(refined + rest, key=lambda x: x.get("distance", float("inf")))

    top = results[: args.top_k]

    out = {
        "passive_mode": bool(args.passive),
        "phase2_passive_policy": bool(args.passive),
        "phase3_micro_raise_pot_build": bool(args.passive),
        "fp_cols_mode": args.fp_cols,
        "chunks_per_candidate": n_chunks,
        "refine_top": args.refine_top,
        "refine_chunks": args.refine_chunks if args.refine_top else 0,
        "stakes_mode": args.stakes,
        "n_fp_dims": len(fp_cols),
        "n_candidates": len(results),
        "top_k": args.top_k,
        "elapsed_seconds": round(elapsed, 1),
        "best_distance": top[0]["distance"] if top else None,
        "worst_kept_distance": top[-1]["distance"] if top else None,
        "top": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {elapsed:.1f}s  best_dist={out['best_distance']:.4f}  "
          f"worst_kept={out['worst_kept_distance']:.4f}  out={args.out}")
    print("\nTop 5 matches:")
    for r in top[:5]:
        c = r["candidate"]
        print(f"  id={r['id']}  dist={r['distance']:.4f}  seats={c['max_seats']}  "
              f"sb/bb={c['sb']}/{c['bb']}  tight={c['profile']['tightness']:.2f} "
              f"aggr={c['profile']['aggression']:.2f} "
              f"pre_def={c['profile']['preflop_defend_bias']:+.2f} "
              f"post_cont={c['profile']['postflop_continue_bias']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
