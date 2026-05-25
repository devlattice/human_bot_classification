"""Repeat match → probe-generate → validate until synthetic chunks fit May-8 bot profile.

Each failed round:
  - Pull fingerprint means toward May-8 hard gold on high-KS features
  - Up-weight those features for the next match
  - Widen LHS search (more candidates)

When KS gates pass (or max rounds), run full generation + train/eval pipeline.

Usage:
  python workspace/hybrid/bot_system/20_iterate_until_may8_profile.py
  MAX_ROUNDS=10 MAX_MEDIAN_KS=0.45 python workspace/hybrid/bot_system/20_iterate_until_may8_profile.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "workspace" / "hybrid" / "bot_system" / "data"
LOG = REPO / "workspace" / "hybrid" / "bot_system" / "logs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "bot_system"))

from may8_validate import profile_reached, validate_vs_may8_hard  # noqa: E402

FP = DATA / "may8_hard_target_fingerprint.json"
GOLD = REPO / "workspace" / "hybrid" / "dataset" / "train" / "gold_features.parquet"
MATCHED = DATA / "may8_reflect_matched_profiles.json"
PROBE_PARQUET = DATA / "may8_reflect_probe.parquet"
GEN_OUT = DATA / "may8_reflect_bot_features.parquet"
RESULTS = DATA / "may8_bot_pipeline_results.json"
BUNDLE = REPO / "workspace" / "hybrid" / "model_bundle_may8_reflect"
ITER_LOG = DATA / "may8_iterate_profile_log.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--max-median-ks", type=float, default=0.45,
                   help="Stop iterating when median KS(gen, may8_bot) <= this.")
    p.add_argument("--max-weighted-median-ks", type=float, default=0.50)
    p.add_argument("--n-candidates-start", type=int, default=80)
    p.add_argument("--n-candidates-step", type=int, default=40)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--probe-top-k", type=int, default=4,
                   help="Profiles used for quick probe generation each round.")
    p.add_argument("--probe-chunks", type=int, default=12,
                   help="Chunks per profile in probe pass.")
    p.add_argument("--full-chunks-per-profile", type=int, default=22)
    p.add_argument("--full-perturb", type=int, default=5)
    p.add_argument("--may8-bot-cap", type=int, default=5000)
    p.add_argument("--mean-pull", type=float, default=0.35,
                   help="Fraction to move fingerprint mean toward gold on bad features.")
    p.add_argument("--skip-full-if-never-pass", action="store_true",
                   help="Exit 2 if profile gate never passed (no full gen).")
    return p.parse_args()


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        print(f"\n>>> {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT)
    tail = log_path.read_text(encoding="utf-8")[-2000:] if log_path.is_file() else ""
    print(tail, flush=True)
    return int(r.returncode)


def adjust_fingerprint(fp: dict, summary: dict, *, mean_pull: float) -> dict:
    """Nudge targets toward May-8 gold on features where probe still mismatches."""
    means = dict(fp.get("feature_means") or {})
    stds = dict(fp.get("feature_stds") or {})
    weights = dict(fp.get("feature_weights") or {})
    pull = float(mean_pull)

    for row in summary.get("features") or []:
        ks = float(row["ks_gen_vs_hard"])
        if ks < 0.30:
            continue
        c = row["feature"]
        if c not in means:
            continue
        target = float(row["hard_mean"])
        means[c] = (1.0 - pull) * float(means[c]) + pull * target
        weights[c] = float(weights.get(c, 1.0)) * (1.0 + 0.15 * ks)
        stds[c] = max(float(stds.get(c, 1e-3)), 1e-3)

    fp["feature_means"] = means
    fp["feature_stds"] = stds
    fp["feature_weights"] = weights
    fp["adjust_round"] = int(fp.get("adjust_round", 0)) + 1
    return fp


def main() -> int:
    args = parse_args()
    py = sys.executable
    LOG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    rc = run(
        [py, "workspace/hybrid/bot_system/06_build_may8_target.py",
         "--hard-only", "--blend", "1.0", "--out", str(FP)],
        LOG / "iter_00_fingerprint.log",
    )
    if rc != 0:
        return rc

    fp = json.loads(FP.read_text(encoding="utf-8"))
    gold = pd.read_parquet(GOLD)
    history: list[dict] = []
    passed_round: int | None = None
    n_cand = args.n_candidates_start

    for rnd in range(1, args.max_rounds + 1):
        t0 = time.time()
        matched_r = DATA / f"may8_reflect_matched_r{rnd}.json"
        seed = 40 + rnd

        rc = run(
            [
                py, "workspace/hybrid/bot_system/03_match_profiles.py",
                "--fp", str(FP),
                "--out", str(matched_r),
                "--passive", "--fp-cols", "may8", "--stakes", "micro",
                "--chunks-per-candidate", "6",
                "--refine-top", "10", "--refine-chunks", "10",
                "--n-candidates", str(n_cand),
                "--top-k", str(args.top_k),
                "--workers", "6",
                "--seed", str(seed),
            ],
            LOG / f"iter_r{rnd}_match.log",
        )
        if rc != 0:
            print(f"[abort] match failed round {rnd}")
            return rc

        matched = json.loads(matched_r.read_text(encoding="utf-8"))
        best_dist = matched.get("best_distance")

        rc = run(
            [
                py, "workspace/hybrid/bot_system/04_generate_targeted_bots.py",
                "--matched", str(matched_r),
                "--out", str(PROBE_PARQUET),
                "--passive", "--source-tag", "may8_matched_bot",
                "--top-k", str(args.probe_top_k),
                "--perturbations-per-seed", "2",
                "--chunks-per-profile", str(args.probe_chunks),
                "--workers", "4",
                "--per-job-timeout", "120",
                "--seed", str(20260500 + rnd),
            ],
            LOG / f"iter_r{rnd}_probe_gen.log",
        )
        if rc != 0:
            print(f"[abort] probe gen failed round {rnd}")
            return rc

        probe = pd.read_parquet(PROBE_PARQUET)
        fp = json.loads(FP.read_text(encoding="utf-8"))
        summary = validate_vs_may8_hard(probe, gold, fp)
        ok = profile_reached(
            summary,
            max_median_ks=args.max_median_ks,
            max_weighted_median_ks=args.max_weighted_median_ks,
        )

        row = {
            "round": rnd,
            "n_candidates": n_cand,
            "best_match_distance": best_dist,
            "n_probe_chunks": int(len(probe)),
            "median_ks": summary.get("median_ks"),
            "weighted_median_ks": summary.get("weighted_median_ks"),
            "mean_ks": summary.get("mean_ks"),
            "profile_pass": ok,
            "elapsed_sec": round(time.time() - t0, 1),
            "worst_3": summary.get("worst_5", [])[-3:],
        }
        history.append(row)
        ITER_LOG.write_text(json.dumps({"rounds": history}, indent=2), encoding="utf-8")

        print(
            f"\n[round {rnd}] median_ks={summary['median_ks']} "
            f"w_median={summary['weighted_median_ks']} "
            f"best_dist={best_dist} pass={ok}"
        )

        if ok:
            passed_round = rnd
            MATCHED.write_bytes(matched_r.read_bytes())
            break

        fp = adjust_fingerprint(fp, summary, mean_pull=args.mean_pull)
        FP.write_text(json.dumps(fp, indent=2), encoding="utf-8")
        n_cand += args.n_candidates_step
        print(f"[round {rnd}] adjusted fingerprint → re-match next round (n_candidates={n_cand})")

    if passed_round is None:
        print(
            f"\n[warn] profile gate not reached after {args.max_rounds} rounds "
            f"(median_ks target {args.max_median_ks})"
        )
        if args.skip_full_if_never_pass:
            return 2
        # use last match for full gen anyway
        last = DATA / f"may8_reflect_matched_r{args.max_rounds}.json"
        if last.is_file():
            MATCHED.write_bytes(last.read_bytes())

    print(f"\n[full gen] using profiles from round {passed_round or 'last'}")
    rc = run(
        [
            py, "workspace/hybrid/bot_system/04_generate_targeted_bots.py",
            "--matched", str(MATCHED),
            "--out", str(GEN_OUT),
            "--passive", "--source-tag", "may8_matched_bot",
            "--top-k", str(args.top_k),
            "--perturbations-per-seed", str(args.full_perturb),
            "--chunks-per-profile", str(args.full_chunks_per_profile),
            "--workers", "6",
            "--per-job-timeout", "120",
            "--seed", "20260508",
        ],
        LOG / "iter_full_gen.log",
    )
    if rc != 0:
        return rc

    run(
        [py, "workspace/hybrid/bot_system/19_validate_may8_generated.py",
         "--generated", str(GEN_OUT), "--fingerprint", str(FP)],
        LOG / "iter_full_validate.log",
    )

    rc = run(
        [
            py, "workspace/hybrid/bot_system/18_eval_may8_bot_pipeline.py",
            "--may8-matched", str(GEN_OUT),
            "--may8-bot-cap", str(args.may8_bot_cap),
            "--bundle-out", str(BUNDLE),
            "--results-out", str(RESULTS),
        ],
        LOG / "iter_full_eval.log",
    )

    out = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "passed_profile_at_round": passed_round,
        "max_median_ks_target": args.max_median_ks,
        "rounds": history,
        "full_generation": str(GEN_OUT),
        "eval_results": str(RESULTS),
    }
    (DATA / "may8_iterate_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[done] iterate summary → {DATA / 'may8_iterate_summary.json'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
