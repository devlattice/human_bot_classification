"""Feature selection v3: filter → pair rescue → dynamic top-X → Optuna → lockbox.

Uses only workspace/hybrid/dataset/train for selection fit and
workspace/hybrid/dataset/test (+ may8_gold_test) for lockbox.

Usage:
  python workspace/hybrid/scripts/feature_selection_v3.py --phase all
  python workspace/hybrid/scripts/feature_selection_v3.py --phase prefilter
  python workspace/hybrid/scripts/feature_selection_v3.py --phase pair_rescue
  python workspace/hybrid/scripts/feature_selection_v3.py --phase search_x --trials-per-x 50
  python workspace/hybrid/scripts/feature_selection_v3.py --phase optuna --trials 300 --pool-x 30
  python workspace/hybrid/scripts/feature_selection_v3.py --phase lockbox --top 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_selection_lib as fsl  # noqa: E402
import train_production_model as tpm  # noqa: E402
from shell_progress import (  # noqa: E402
    iter_progress,
    optuna_progress_callback,
    phase_banner,
)

HYBRID_DIR = REPO_ROOT / "workspace" / "hybrid"
OUT_PREFILTER = HYBRID_DIR / "feature_selection_prefilter.json"
OUT_RANKED = HYBRID_DIR / "feature_selection_ranked.json"
OUT_PAIR = HYBRID_DIR / "feature_selection_pair_rescue.json"
OUT_X_CURVE = HYBRID_DIR / "feature_selection_x_curve.json"
OUT_CANDIDATES = HYBRID_DIR / "selected_features_v3_candidates.json"
OUT_FINAL = HYBRID_DIR / "selected_features_v3.json"
OUT_REPORT = HYBRID_DIR / "feature_selection_v3_report.txt"

COARSE_X_GRID = [15, 25, 35, 45]
FULL_X_GRID = [10, 15, 20, 25, 30, 35, 40]


def _load_state() -> dict:
    gold = fsl.load_gold_train()
    candidates = fsl.prefilter_candidates(gold)
    ctx = fsl.SelectionContext(gold, candidates)
    ranked: list[str] = []
    pool: list[str] = list(candidates)
    best_x = 30

    if OUT_RANKED.is_file():
        ranked = json_load(OUT_RANKED).get("ranked", [])
    if OUT_PAIR.is_file():
        data = json_load(OUT_PAIR)
        pool = data.get("pool", pool)
        best_x = data.get("suggested_pool_x", best_x)
    if OUT_X_CURVE.is_file():
        xc = json_load(OUT_X_CURVE)
        best_x = int(xc.get("best_x", best_x))

    return {
        "gold": gold,
        "candidates": candidates,
        "ctx": ctx,
        "ranked": ranked,
        "pool": pool,
        "best_x": best_x,
    }


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def phase_prefilter(args: argparse.Namespace) -> None:
    gold = fsl.load_gold_train()
    raw = fsl.prefilter_candidates(gold)
    datasets_probe = fsl.load_train_tables(raw)
    candidates = fsl.intersect_train_columns(raw, datasets_probe)
    print(
        f"[prefilter] gold rows={len(gold)}  raw={len(raw)}  "
        f"after intersect(gold,zenodo,acpc)={len(candidates)}"
    )

    ctx = fsl.SelectionContext(gold, candidates, seed=args.seed)
    rank_list = candidates
    if args.max_rank and args.max_rank < len(candidates):
        rank_list = candidates[: args.max_rank]
        print(f"[prefilter] --max-rank {args.max_rank} (truncated for speed)")

    print("[prefilter] univariate LOOCV ranking (single-feature models)...", flush=True)
    ranked_scores = fsl.rank_univariate(
        ctx, rank_list, fast=not args.full_rf, quiet=args.quiet
    )
    # append unranked tail in stable order
    ranked_feats = [f for f, _ in ranked_scores]
    for c in candidates:
        if c not in ranked_feats:
            ranked_scores.append((c, 0.0))
    ranked_scores.sort(key=lambda x: -x[1])
    ranked = fsl.build_diverse_ranked(ranked_scores, gold)

    fsl.save_json(
        OUT_PREFILTER,
        {
            "n_candidates": len(candidates),
            "n_raw_prefilter": len(raw),
            "candidates": candidates,
            "rank_objective": "0.50*min_gold_loocv + 0.30*mean_gold_loocv + 0.15*acpc_recall - 0.20*max_fpr",
            "fit_data": "train/gold_loocv_folds + train/zenodo,public,full_spectrum,acpc",
        },
    )
    fsl.save_json(
        OUT_RANKED,
        {
            "ranked": ranked,
            "top20": [{"feature": f, "score": round(s, 4)} for f, s in ranked_scores[:20]],
        },
    )
    print(f"[prefilter] saved {OUT_PREFILTER}")
    print(f"[prefilter] saved {OUT_RANKED} ({len(ranked)} ranked)")


def phase_pair_rescue(args: argparse.Namespace) -> None:
    st = _load_state()
    if not st["ranked"]:
        print("[pair_rescue] missing ranked list — run --phase prefilter first")
        sys.exit(1)

    result = fsl.run_pair_rescue(
        st["ctx"],
        st["ranked"],
        x_base=args.x_base,
        borderline=args.borderline,
        fast=not args.full_rf,
        quiet=args.quiet,
    )
    result["suggested_pool_x"] = args.x_base
    fsl.save_json(OUT_PAIR, result)
    print(f"[pair_rescue] core={len(result['core'])}  promoted={len(result['promoted'])}")
    print(f"[pair_rescue] pool size={len(result['pool'])}  saved {OUT_PAIR}")


def _optuna_objective(
    trial: optuna.Trial,
    ctx: fsl.SelectionContext,
    pool: list[str],
    *,
    min_features: int,
    max_features: int,
    fast: bool,
    pool_x: int | None,
) -> float:
    if len(pool) > 45:
        pool = pool[:45]

    selected: list[str] = []
    for f in pool:
        if trial.suggest_int(f"inc_{f}", 0, 1):
            selected.append(f)

    if len(selected) < min_features or len(selected) > max_features:
        return -1e6

    kw = fsl.default_rf_kwargs(ctx.seed, fast=fast)
    metrics = fsl.evaluate_feature_subset(ctx, selected, kw)
    if not metrics.get("valid"):
        return -1e6

    trial.set_user_attr("features", selected)
    trial.set_user_attr("metrics", {k: v for k, v in metrics.items() if k != "per_day_recall"})

    return fsl.composite_score(metrics, pool_penalty_x=pool_x)


def phase_search_x(args: argparse.Namespace) -> None:
    st = _load_state()
    ranked = st["ranked"] or st["candidates"]
    ctx = st["ctx"]
    grid = COARSE_X_GRID if args.coarse_only else FULL_X_GRID

    curve: list[dict] = []
    best_x = grid[0]
    best_median = -1e9

    show_bar = not args.quiet
    cb = optuna_progress_callback(every=max(5, args.trials_per_x // 10), disable=args.quiet)

    for x in iter_progress(grid, desc="[search_x] pool sizes", disable=args.quiet):
        pool = ranked[: min(x, len(ranked))]
        print(f"\n[search_x] X={x}  pool={len(pool)}  trials={args.trials_per_x}", flush=True)

        study = optuna.create_study(
            direction="maximize",
            study_name=f"fs_v3_x{x}",
            sampler=optuna.samplers.TPESampler(seed=args.seed + x),
        )

        def obj(trial: optuna.Trial) -> float:
            return _optuna_objective(
                trial,
                ctx,
                pool,
                min_features=args.min_features,
                max_features=args.max_features,
                fast=not args.full_rf,
                pool_x=x,
            )

        study.optimize(
            obj,
            n_trials=args.trials_per_x,
            show_progress_bar=show_bar,
            callbacks=[cb],
        )

        vals = [t.value for t in study.trials if t.value is not None and t.value > -1e5]
        median = float(np.median(vals)) if vals else -1e9
        best = float(study.best_value) if study.best_trial else -1e9
        curve.append({"x": x, "best": best, "median": median, "n_ok": len(vals)})
        print(f"  X={x}  best={best:.4f}  median={median:.4f}  ok_trials={len(vals)}")

        if median > best_median:
            best_median = median
            best_x = x

    # Fine grid around winner
    if not args.coarse_only:
        fine = sorted(
            {max(10, best_x - 10), max(10, best_x - 5), best_x, min(45, best_x + 5), min(45, best_x + 10)}
        )
        for x in iter_progress(fine, desc="[search_x] fine grid", disable=args.quiet):
            if x in [c["x"] for c in curve]:
                continue
            pool = ranked[: min(x, len(ranked))]
            print(f"[search_x] fine X={x}  pool={len(pool)}", flush=True)
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
            )
            study.optimize(
                lambda t, px=x, pl=pool: _optuna_objective(
                    t,
                    ctx,
                    pl,
                    min_features=args.min_features,
                    max_features=args.max_features,
                    fast=not args.full_rf,
                    pool_x=px,
                ),
                n_trials=max(30, args.trials_per_x // 2),
                show_progress_bar=show_bar,
                callbacks=[cb],
            )
            vals = [t.value for t in study.trials if t.value is not None and t.value > -1e5]
            median = float(np.median(vals)) if vals else -1e9
            curve.append({"x": x, "best": float(study.best_value), "median": median, "n_ok": len(vals)})
            if median > best_median:
                best_median = median
                best_x = x

    out = {"curve": curve, "best_x": best_x, "best_median": best_median}
    fsl.save_json(OUT_X_CURVE, out)

    # merge into pair rescue json if present
    if OUT_PAIR.is_file():
        pr = json_load(OUT_PAIR)
        pr["best_x"] = best_x
        fsl.save_json(OUT_PAIR, pr)

    print(f"\n[search_x] best_x={best_x}  saved {OUT_X_CURVE}")


def phase_optuna(args: argparse.Namespace) -> None:
    st = _load_state()
    ranked = st["ranked"] or st["candidates"]
    pool_full = st["pool"] if st["pool"] else ranked
    x = args.pool_x or st["best_x"]
    pool = pool_full[: min(x, len(pool_full))]
    ctx = st["ctx"]

    print(f"[optuna] pool_x={x}  pool_len={len(pool)}  trials={args.trials}", flush=True)
    cb = optuna_progress_callback(every=max(10, args.trials // 20), disable=args.quiet)

    study = optuna.create_study(
        direction="maximize",
        study_name="fs_v3_final",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # warm-start from ROBUST_FEATURES
    robust = [f for f in tpm.ROBUST_FEATURES if f in pool]
    if len(robust) >= args.min_features:
        m0 = fsl.evaluate_feature_subset(
            ctx, robust, fsl.default_rf_kwargs(args.seed, fast=not args.full_rf)
        )
        study.enqueue_trial(
            {f"inc_{f}": int(f in robust) for f in pool},
        )
        print(f"[optuna] warm-start ROBUST subset n={len(robust)}  score={m0.get('score', 0):.4f}")

    study.optimize(
        lambda t: _optuna_objective(
            t,
            ctx,
            pool,
            min_features=args.min_features,
            max_features=args.max_features,
            fast=not args.full_rf,
            pool_x=x,
        ),
        n_trials=args.trials,
        show_progress_bar=not args.quiet,
        callbacks=[cb],
    )

    trials_out = []
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > -1e5],
        key=lambda t: t.value,
        reverse=True,
    )
    for t in sorted_trials[: args.top_save]:
        trials_out.append({
            "value": float(t.value),
            "features": t.user_attrs.get("features", []),
            "metrics": t.user_attrs.get("metrics", {}),
        })

    fsl.save_json(
        OUT_CANDIDATES,
        {
            "pool_x": x,
            "pool": pool,
            "best_value": float(study.best_value) if study.best_trial else None,
            "best_features": study.best_trial.user_attrs.get("features", []) if study.best_trial else [],
            "trials": trials_out,
        },
    )
    print(f"[optuna] best={study.best_value:.4f}  saved {OUT_CANDIDATES}")


def phase_lockbox(args: argparse.Namespace) -> None:
    if not OUT_CANDIDATES.is_file():
        print("[lockbox] run --phase optuna first")
        sys.exit(1)

    cand = json_load(OUT_CANDIDATES)
    gold = fsl.load_gold_train()
    finalists = cand.get("trials", [])[: args.top]
    if not finalists:
        print("[lockbox] no trials in candidates file")
        sys.exit(1)

    kw = tpm.rf_kwargs_from_namespace(
        argparse.Namespace(
            rf_n_estimators=300,
            rf_max_depth="6",
            rf_min_samples_leaf=15,
            rf_min_samples_split=2,
            rf_max_features="sqrt",
            rf_max_samples=1.0,
            rf_class_weight="balanced",
            rf_ccp_alpha=0.0,
        ),
        args.seed,
    )

    report_lines: list[str] = []
    lockbox_results: list[dict] = []

    for i, fin in enumerate(
        iter_progress(finalists, desc="[lockbox] finalists", disable=args.quiet)
    ):
        features = fin["features"]
        print(f"\n[lockbox] finalist {i + 1}/{len(finalists)}  n_feat={len(features)}", flush=True)

        datasets = tpm.load_datasets(features)
        X_raw, y, _ = tpm.build_training_data(datasets, features, args.seed)
        X_t, transform_meta = tpm.fit_transform_pipeline(X_raw, features)
        rf = RandomForestClassifier(**kw)
        rf.fit(X_t, y)

        block = {"features": features, "n_features": len(features), "tests": {}}

        tests = [
            ("may8_gold_test", tpm.MAY8_GOLD_TEST_PATH),
            ("zenodo_test", tpm.TEST_DIR / "zenodo_test_features.parquet"),
            ("public_test", tpm.TEST_DIR / "public_test_features.parquet"),
            ("acpc_bot_test", tpm.TEST_DIR / "acpc_bot_test_features.parquet"),
        ]
        for name, path in tests:
            if path.is_file():
                block["tests"][name] = fsl.eval_test_parquet(path, features, rf, transform_meta)

        lockbox_results.append(block)
        may8 = block["tests"].get("may8_gold_test", {})
        report_lines.append(
            f"#{i + 1} n={len(features)} may8_recall={may8.get('bot_recall_pct')}% "
            f"may8_fpr={may8.get('human_fpr_pct')}%"
        )
        print(f"  may8: {may8}")

    def pick_winner(rows: list[dict]) -> int:
        best_i = 0
        best_rec = -1.0
        for i, row in enumerate(rows):
            m8 = row["tests"].get("may8_gold_test", {})
            rec = m8.get("bot_recall_pct")
            if rec is None:
                continue
            zen = row["tests"].get("zenodo_test", {})
            pub = row["tests"].get("public_test", {})
            fpr_z = zen.get("human_fpr_pct", 100) or 100
            fpr_p = pub.get("human_fpr_pct", 100) or 100
            if fpr_z > 1.0 or fpr_p > 1.0:
                continue
            if rec > best_rec:
                best_rec = rec
                best_i = i
        return best_i

    win_i = pick_winner(lockbox_results)
    winner = lockbox_results[win_i]
    win_features = winner["features"]

    final = {
        "version": 3,
        "selected_features": win_features,
        "winner_index": win_i,
        "pool_x": cand.get("pool_x"),
        "lockbox": lockbox_results,
        "baseline_robust_54": list(tpm.ROBUST_FEATURES),
    }
    fsl.save_json(OUT_FINAL, final)

    report_lines = [
        "FEATURE SELECTION V3 — LOCKBOX REPORT",
        f"Winner: finalist #{win_i + 1}  ({len(win_features)} features)",
        "",
        *report_lines,
        "",
        f"Features: {win_features}",
    ]
    OUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n[lockbox] winner #{win_i + 1}  saved {OUT_FINAL}")
    print(f"[lockbox] report {OUT_REPORT}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature selection v3")
    p.add_argument(
        "--phase",
        choices=("all", "prefilter", "pair_rescue", "search_x", "optuna", "lockbox"),
        default="all",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x-base", type=int, default=30)
    p.add_argument("--borderline", type=int, default=25)
    p.add_argument("--trials-per-x", type=int, default=50)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--pool-x", type=int, default=None)
    p.add_argument("--min-features", type=int, default=5)
    p.add_argument("--max-features", type=int, default=25)
    p.add_argument("--top", type=int, default=10, help="Lockbox finalists")
    p.add_argument("--top-save", type=int, default=20)
    p.add_argument("--coarse-only", action="store_true")
    p.add_argument("--full-rf", action="store_true", help="Slower 300-tree RF in inner loop")
    p.add_argument("--max-rank", type=int, default=None, help="Only LOOCV-rank first N candidates (smoke)")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal progress (no tqdm bars; fewer trial logs)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    phases = (
        ["prefilter", "pair_rescue", "search_x", "optuna", "lockbox"]
        if args.phase == "all"
        else [args.phase]
    )

    for i, ph in enumerate(phases, start=1):
        phase_banner(ph, i, len(phases))
        t_ph = time.time()
        if ph == "prefilter":
            phase_prefilter(args)
        elif ph == "pair_rescue":
            phase_pair_rescue(args)
        elif ph == "search_x":
            phase_search_x(args)
        elif ph == "optuna":
            phase_optuna(args)
        elif ph == "lockbox":
            phase_lockbox(args)
        print(f"[{ph}] finished in {time.time() - t_ph:.1f}s", flush=True)

    print(f"\n[done] total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
