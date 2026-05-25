#!/usr/bin/env python3
"""
Versioned end-to-end DANN automation from Parquet paths.

Runs:
1) Labeled/unlabeled Parquet → npz export
2) ``hparam_grid_search.py`` → ``train_dann.py`` per trial
3) Retrain ``train_dann.py`` with best hyperparameters
4) For each test parquet: ``export_parquet_to_target_npz`` → ``infer_dann.py`` → ``eval_dann_holdout.py``
5) ``summary.csv``, ``run_manifest.json``, optional compare vs previous version
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "workspace" / "DANN" / "scripts"
EXPORT_SOURCE = SCRIPTS_DIR / "export_parquet_to_source_npz.py"
EXPORT_TARGET = SCRIPTS_DIR / "export_parquet_to_target_npz.py"
GRID_SEARCH = SCRIPTS_DIR / "hparam_grid_search.py"
TRAIN = SCRIPTS_DIR / "train_dann.py"
INFER = SCRIPTS_DIR / "infer_dann.py"
EVAL = SCRIPTS_DIR / "eval_dann_holdout.py"


def _with_dbf_parquet(path: Path) -> Path:
    """``train.parquet`` -> ``train_with_dbf.parquet`` in the same directory (idempotent)."""
    p = Path(path)
    if p.suffix.lower() != ".parquet":
        return p
    stem = p.stem
    if stem.endswith("_with_dbf"):
        return p
    return p.with_name(f"{stem}_with_dbf.parquet")


def _default_tests(*, use_dbf: bool) -> list[Path]:
    """Built-in test parquets under feature_2/data (names match on-disk layout).

    With ``use_dbf``, the fifth file is ``irc_val`` (not ``irc_train``): a
    ``irc_train_with_dbf.parquet`` companion is not always present, while
    ``irc_val_with_dbf.parquet`` is generated alongside ``irc_val.parquet``.
    """
    base = REPO_ROOT / "workspace" / "preprocess" / "statistical_test" / "explorer" / "feature_2" / "data"
    out = [
        base / "test" / "pb_1.parquet",
        base / "test" / "pb_2.parquet",
        base / "test" / "holdout_1.parquet",
        base / "test" / "holdout_2.parquet",
    ]
    out.append(base / "irc" / ("irc_val.parquet" if use_dbf else "irc_train.parquet"))
    return out


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "workspace" / "preprocess" / "statistical_test" / "explorer" / "feature_2" / "data"
    p = argparse.ArgumentParser(description="Run versioned DANN train/eval automation.")
    p.add_argument(
        "--train-parquet",
        type=Path,
        default=base / "public" / "train.parquet",
    )
    p.add_argument(
        "--val-parquet",
        type=Path,
        default=base / "public" / "val.parquet",
    )
    p.add_argument(
        "--validator-parquet",
        type=Path,
        default=base / "validator" / "validator.parquet",
    )
    p.add_argument(
        "--test-parquet",
        action="append",
        type=Path,
        default=None,
        help="Repeatable. If omitted, built-in defaults are used.",
    )
    p.add_argument(
        "--tune-parquet",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional labeled cross-domain validation parquet(s) used during hparam/retrain "
            "selection only (multi-objective tuning). Repeatable."
        ),
    )
    p.add_argument("--label-col", default="label")
    p.add_argument(
        "--grid-json",
        type=Path,
        default=REPO_ROOT / "workspace" / "DANN" / "scripts" / "hparam_grid.json",
    )
    p.add_argument("--grid-epochs", type=int, default=40)
    p.add_argument(
        "--retrain-epochs",
        type=int,
        default=80,
        help="Final retrain epochs using best grid-search hyperparameters.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "workspace" / "DANN" / "artifacts" / "auto",
    )
    p.add_argument(
        "--infer-root",
        type=Path,
        default=REPO_ROOT / "workspace" / "DANN" / "infer",
    )
    p.add_argument("--version-prefix", default="v")
    p.add_argument(
        "--cleanup-trials",
        action="store_true",
        help="Delete hparam_trials after the run (runs even if a later step fails).",
    )
    p.add_argument(
        "--val-selection-metric",
        type=str,
        default="bot_recall_at_human_fpr_cap",
        choices=(
            "val_acc",
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ),
        help="Validation objective for checkpoint pick + grid ranking (forwarded to train/grid).",
    )
    p.add_argument(
        "--target-human-fpr",
        type=float,
        default=0.05,
        help="Cap on human FPR when tuning threshold on source val.",
    )
    p.add_argument("--threshold-grid-size", type=int, default=401)
    p.add_argument("--threshold-tie-ref", type=float, default=0.5)
    p.add_argument(
        "--eval-threshold",
        type=float,
        default=None,
        help="If set, overrides checkpoint meta selected_threshold in eval_dann_holdout.",
    )
    p.add_argument(
        "--use-dbf",
        action="store_true",
        help=(
            "Resolve inputs to *_with_dbf.parquet beside each base name: train/val/validator "
            "and each default or explicit --test-parquet path."
        ),
    )
    p.add_argument(
        "--labeled-seat-col",
        type=str,
        default=None,
        help="If set, labeled npz exports include seat_bucket (e.g. n_players_max) for hybrid nuisance training.",
    )
    p.add_argument(
        "--hybrid-nuisance-seat",
        action="store_true",
        help="Enable seat-bucket nuisance head in train_dann (requires --labeled-seat-col exports).",
    )
    p.add_argument("--hybrid-nuisance-weight", type=float, default=0.5)
    p.add_argument("--n-seat-buckets", type=int, default=9)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[run_dann_auto] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _export_source_cmd(parquet: Path, out_npz: Path, label_col: str, seat_col: str | None) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(EXPORT_SOURCE),
        "--parquet",
        str(parquet),
        "--out-npz",
        str(out_npz),
        "--label-col",
        label_col,
    ]
    if seat_col:
        cmd.extend(["--seat-bucket-col", str(seat_col)])
    return cmd


def _next_version_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_n = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return root / f"{prefix}{max_n + 1:04d}"


def _trial_sort_key(row: dict) -> tuple[float, ...]:
    """Larger tuple compares greater (we sort descending)."""
    metric = (row.get("val_selection_metric") or "val_acc").strip()
    try:
        score = float(row.get("best_score", row.get("best_val_acc", "nan")))
    except Exception:
        return (float("-inf"),)
    if score != score:
        return (float("-inf"),)
    if metric == "val_acc":
        return (score,)
    try:
        feas = float(row.get("val_threshold_feasible", 1) or 1)
    except Exception:
        feas = 1.0
    try:
        hf = float(row.get("val_human_fpr_at_selected_threshold", "nan"))
    except Exception:
        hf = float("nan")
    if hf != hf:
        hf = 1.0
    if metric in (
        "bot_recall_at_human_fpr_cap_then_domain_confusion",
        "multi_objective_generalization",
    ):
        try:
            dloss = float(row.get("domain_confusion_loss_source_val_vs_target", "nan"))
        except Exception:
            dloss = float("nan")
        if dloss != dloss:
            dloss = float("-inf")
        return (feas, score, -hf, dloss)
    return (feas, score, -hf)


def _select_best_trial(csv_path: Path) -> dict:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in grid result: {csv_path}")
    valid = []
    for r in rows:
        k = _trial_sort_key(r)
        if k[0] == k[0] and k[0] > float("-inf"):
            valid.append((k, r))
    if not valid:
        raise SystemExit("Grid search produced no valid best_score / best_val_acc rows.")
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


def _threshold_for_eval(ckpt: Path, fallback: float = 0.5) -> float:
    meta_path = ckpt.with_suffix(".meta.json")
    if not meta_path.is_file():
        return fallback
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        return float(meta.get("selected_threshold", fallback))
    except Exception:
        return fallback


def _print_checkpoint_selection_meta(ckpt: Path, *, title: str) -> None:
    """Print val / multi-objective selection summary from ``train_dann`` sidecar meta."""
    meta_path = ckpt.with_suffix(".meta.json")
    if not meta_path.is_file():
        print(f"[run_dann_auto] {title}: no meta file {meta_path}", flush=True)
        return
    meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    metric = str(meta.get("val_selection_metric", "")).strip()
    print(f"[run_dann_auto] === {title}  ({meta_path.name}) ===", flush=True)
    print(f"  val_selection_metric={metric}", flush=True)
    cap = meta.get("target_human_fpr", "")
    th = meta.get("selected_threshold", "")
    feas = meta.get("val_threshold_feasible", "")
    vbr = meta.get("val_bot_recall_at_selected_threshold", "")
    vhf = meta.get("val_human_fpr_at_selected_threshold", "")
    bscore = meta.get("best_score", "")
    print(
        f"  human_fpr_cap={cap}  selected_threshold={th}  val_feasible={feas}  "
        f"val_bot_recall={vbr}  val_human_fpr={vhf}  best_score={bscore}",
        flush=True,
    )
    if metric == "multi_objective_generalization":
        dloss = meta.get("domain_confusion_loss_source_val_vs_target", "")
        dacc = meta.get("domain_confusion_acc_source_val_vs_target", "")
        print(f"  domain_confusion (source val vs target): loss={dloss}  acc={dacc}", flush=True)
        rows: List[Dict[str, Any]] = list(meta.get("extra_val_metrics") or [])
        if not rows:
            print("  per-domain: (no extra_val_metrics in meta)", flush=True)
        else:
            print(
                "  per-domain (each row: FPR-capped threshold sweep on that domain; "
                "threshold can differ per domain):",
                flush=True,
            )
            brs: List[float] = []
            all_feas = True
            for r in rows:
                name = str(r.get("name", "?"))
                rf = bool(r.get("feasible", False))
                all_feas = all_feas and rf
                br = float(r.get("bot_recall", 0.0))
                brs.append(br)
                hf = float(r.get("human_fpr", 1.0))
                tt = float(r.get("threshold", 0.5))
                print(
                    f"    {name}: feasible={rf}  human_fpr={hf:.6f}  bot_recall={br:.6f}  threshold={tt:.6f}",
                    flush=True,
                )
            worst = min(brs) if brs else 0.0
            mean_br = sum(brs) / len(brs) if brs else 0.0
            print(
                f"  multi-objective aggregate: worst_bot_recall={worst:.6f}  mean_bot_recall={mean_br:.6f}  "
                f"all_domains_feasible={all_feas}",
                flush=True,
            )
    print(f"[run_dann_auto] === end {title} ===", flush=True)


def _dataset_name(path: Path) -> str:
    """Stable unique folder name for each test parquet.

    ``path.stem`` alone collides when two files share a name (e.g. both
    ``holdout_1.parquet`` under ``holdout/rb_B/`` vs ``holdout_zenodo/rb_B/``).
    Use the last two parent directory names plus the stem.
    """
    p = path.expanduser().resolve()
    return f"{p.parent.parent.name}__{p.parent.name}__{p.stem}"


def _conf_to_rates(cm: list[list[int]]) -> tuple[float, float]:
    tn, fp = cm[0]
    fn, tp = cm[1]
    fpr = float(fp) / float(tn + fp) if (tn + fp) > 0 else 0.0
    bot_recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    return fpr, bot_recall


def _resolve_test_parquets(arg_paths: list[Path] | None, use_dbf: bool) -> list[Path]:
    """Resolve and verify all test parquets exist (call before grid search)."""
    if arg_paths is not None:
        paths = list(arg_paths)
    else:
        paths = [_with_dbf_parquet(p) if use_dbf else p for p in _default_tests(use_dbf=use_dbf)]
    out: list[Path] = []
    for p in paths:
        rp = Path(p).expanduser().resolve()
        if not rp.is_file():
            raise SystemExit(f"Missing test parquet: {rp}")
        out.append(rp)
    return out


def main() -> int:
    args = parse_args()
    if args.use_dbf:
        args.train_parquet = _with_dbf_parquet(args.train_parquet)
        args.val_parquet = _with_dbf_parquet(args.val_parquet)
        args.validator_parquet = _with_dbf_parquet(args.validator_parquet)
        if args.test_parquet:
            args.test_parquet = [_with_dbf_parquet(p) for p in args.test_parquet]

    train_parquet = args.train_parquet.expanduser().resolve()
    val_parquet = args.val_parquet.expanduser().resolve()
    validator_parquet = args.validator_parquet.expanduser().resolve()
    grid_json = args.grid_json.expanduser().resolve()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    infer_root = args.infer_root.expanduser().resolve()
    versions_root = infer_root / "versions"

    for p in [train_parquet, val_parquet, validator_parquet, grid_json]:
        if not p.is_file():
            raise SystemExit(f"Missing required input: {p}")

    test_paths = _resolve_test_parquets(args.test_parquet, args.use_dbf)

    run_dir = _next_version_dir(versions_root, args.version_prefix)
    prev_dir = None
    if run_dir.name != f"{args.version_prefix}0001":
        prev_num = int(run_dir.name.replace(args.version_prefix, "")) - 1
        prev_dir = versions_root / f"{args.version_prefix}{prev_num:04d}"

    run_dir.mkdir(parents=True, exist_ok=True)
    run_art = artifacts_root / run_dir.name
    run_art.mkdir(parents=True, exist_ok=True)

    source_train_npz = run_art / "source_train.npz"
    source_val_npz = run_art / "source_val.npz"
    target_train_npz = run_art / "target_train.npz"
    extra_val_npzs: list[Path] = []
    feat_json = run_art / "source_train.feature_columns.json"
    trials_dir = run_art / "hparam_trials"
    grid_csv = run_art / "hparam_results.csv"

    seat_col = str(args.labeled_seat_col).strip() if args.labeled_seat_col else None
    if bool(args.hybrid_nuisance_seat) and not seat_col:
        raise SystemExit("--hybrid-nuisance-seat requires --labeled-seat-col (e.g. n_players_max)")
    try:
        _run(_export_source_cmd(train_parquet, source_train_npz, args.label_col, seat_col))
        _run(_export_source_cmd(val_parquet, source_val_npz, args.label_col, seat_col))
        _run(
            [
                sys.executable,
                str(EXPORT_TARGET),
                "--parquet",
                str(validator_parquet),
                "--feature-columns-json",
                str(feat_json),
                "--out-npz",
                str(target_train_npz),
            ]
        )
        for i, tp in enumerate(args.tune_parquet):
            tune_parquet = Path(tp).expanduser().resolve()
            if not tune_parquet.is_file():
                raise SystemExit(f"Missing tune parquet: {tune_parquet}")
            tune_npz = run_art / f"tune_val_{i:02d}.npz"
            _run(_export_source_cmd(tune_parquet, tune_npz, args.label_col, seat_col))
            extra_val_npzs.append(tune_npz)
        _run(
            [
                sys.executable,
                str(GRID_SEARCH),
                "--source-npz",
                str(source_train_npz),
                "--source-val-npz",
                str(source_val_npz),
                "--target-npz",
                str(target_train_npz),
                "--grid-json",
                str(grid_json),
                "--epochs",
                str(args.grid_epochs),
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--trials-dir",
                str(trials_dir),
                "--out-csv",
                str(grid_csv),
            ]
            + (["--device", args.device] if args.device else [])
            + sum([["--extra-val-npz", str(p)] for p in extra_val_npzs], [])
            + [
                "--val-selection-metric",
                str(args.val_selection_metric),
                "--target-human-fpr",
                str(args.target_human_fpr),
                "--threshold-grid-size",
                str(args.threshold_grid_size),
                "--threshold-tie-ref",
                str(args.threshold_tie_ref),
            ]
            + (["--hybrid-nuisance-seat"] if args.hybrid_nuisance_seat else [])
            + (
                [
                    "--hybrid-nuisance-weight",
                    str(args.hybrid_nuisance_weight),
                    "--n-seat-buckets",
                    str(args.n_seat_buckets),
                ]
                if args.hybrid_nuisance_seat
                else []
            )
        )

        best = _select_best_trial(grid_csv)
        best_ckpt = Path(best["ckpt"]).expanduser().resolve()
        if not best_ckpt.is_file():
            raise SystemExit(f"Best trial checkpoint missing: {best_ckpt}")
        _print_checkpoint_selection_meta(best_ckpt, title="grid_best_trial")

        selected_ckpt = run_art / "dann_best_hparams.pt"
        retrain_cmd = [
            sys.executable,
            str(TRAIN),
            "--source-npz",
            str(source_train_npz),
            "--source-val-npz",
            str(source_val_npz),
            "--target-npz",
            str(target_train_npz),
            "--out",
            str(selected_ckpt),
            "--epochs",
            str(args.retrain_epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
        ]
        for k in ("lambda_max", "lambda_gamma", "lr", "dropout", "weight_decay", "hidden_dim", "feat_dim"):
            if k in best and str(best[k]).strip() != "":
                retrain_cmd.extend([f"--{k.replace('_', '-')}", str(best[k])])
        retrain_cmd.extend(
            [
                "--val-selection-metric",
                str(args.val_selection_metric),
                "--target-human-fpr",
                str(args.target_human_fpr),
                "--threshold-grid-size",
                str(args.threshold_grid_size),
                "--threshold-tie-ref",
                str(args.threshold_tie_ref),
            ]
        )
        if args.hybrid_nuisance_seat:
            retrain_cmd.append("--hybrid-nuisance-seat")
            retrain_cmd.extend(
                [
                    "--hybrid-nuisance-weight",
                    str(args.hybrid_nuisance_weight),
                    "--n-seat-buckets",
                    str(args.n_seat_buckets),
                ]
            )
        for p in extra_val_npzs:
            retrain_cmd.extend(["--extra-val-npz", str(p)])
        if args.device:
            retrain_cmd.extend(["--device", args.device])
        _run(retrain_cmd)
        _print_checkpoint_selection_meta(selected_ckpt, title="final_retrain_checkpoint")

        grid_best_ckpt = run_art / "grid_best_trial.pt"
        shutil.copy2(best_ckpt, grid_best_ckpt)
        best_meta = best_ckpt.with_suffix(".meta.json")
        if best_meta.is_file():
            shutil.copy2(best_meta, grid_best_ckpt.with_suffix(".meta.json"))

        eval_thr = (
            float(args.eval_threshold)
            if args.eval_threshold is not None
            else _threshold_for_eval(selected_ckpt)
        )

        rows: List[dict] = []
        for test_parquet in test_paths:
            name = _dataset_name(test_parquet)
            out_ds = run_dir / name
            out_ds.mkdir(parents=True, exist_ok=True)
            test_npz = out_ds / f"{name}.npz"
            probs_npz = out_ds / "query_probs.npz"
            eval_out = out_ds / "test"
            _run(
                [
                    sys.executable,
                    str(EXPORT_TARGET),
                    "--parquet",
                    str(test_parquet),
                    "--feature-columns-json",
                    str(feat_json),
                    "--out-npz",
                    str(test_npz),
                ]
            )
            _run(
                [
                    sys.executable,
                    str(INFER),
                    "--ckpt",
                    str(selected_ckpt),
                    "--npz",
                    str(test_npz),
                    "--out-npz",
                    str(probs_npz),
                ]
                + (["--device", args.device] if args.device else [])
            )
            _run(
                [
                    sys.executable,
                    str(EVAL),
                    "--probs-npz",
                    str(probs_npz),
                    "--labels-parquet",
                    str(test_parquet),
                    "--features-npz",
                    str(test_npz),
                    "--out-dir",
                    str(eval_out),
                    "--label-col",
                    args.label_col,
                    "--threshold",
                    str(eval_thr),
                ]
            )
            metrics = json.loads((eval_out / "metrics.json").read_text(encoding="utf-8"))
            cm = metrics["confusion_matrix"]
            fpr, bot_recall = _conf_to_rates(cm)
            rows.append(
                {
                    "dataset": name,
                    "n": int(metrics["n"]),
                    "accuracy": float(metrics["accuracy"]),
                    "roc_auc": float(metrics["roc_auc"]),
                    "average_precision": float(metrics["average_precision"]),
                    "fpr": fpr,
                    "bot_recall": bot_recall,
                }
            )

        summary_csv = run_dir / "summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["dataset", "n", "accuracy", "roc_auc", "average_precision", "fpr", "bot_recall"],
            )
            w.writeheader()
            w.writerows(rows)

        manifest = {
            "version": run_dir.name,
            "train_parquet": str(train_parquet),
            "val_parquet": str(val_parquet),
            "validator_parquet": str(validator_parquet),
            "use_dbf": bool(args.use_dbf),
            "test_parquets": [str(p) for p in test_paths],
            "tune_parquets": [str(Path(p).expanduser().resolve()) for p in args.tune_parquet],
            "labeled_seat_col": seat_col,
            "hybrid_nuisance_seat": bool(args.hybrid_nuisance_seat),
            "hybrid_nuisance_weight": float(args.hybrid_nuisance_weight),
            "n_seat_buckets": int(args.n_seat_buckets),
            "label_col": args.label_col,
            "grid_json": str(grid_json),
            "grid_epochs": args.grid_epochs,
            "retrain_epochs": args.retrain_epochs,
            "val_selection_metric": str(args.val_selection_metric),
            "target_human_fpr": float(args.target_human_fpr),
            "threshold_grid_size": int(args.threshold_grid_size),
            "threshold_tie_ref": float(args.threshold_tie_ref),
            "eval_threshold": float(eval_thr),
            "eval_threshold_override": args.eval_threshold,
            "best_trial": best,
            "grid_best_trial_ckpt": str(grid_best_ckpt),
            "selected_ckpt": str(selected_ckpt),
            "summary_csv": str(summary_csv),
            "cleanup_trials": bool(args.cleanup_trials),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if prev_dir and (prev_dir / "summary.csv").is_file():
            with (prev_dir / "summary.csv").open("r", encoding="utf-8", newline="") as f:
                prev_rows = {r["dataset"]: r for r in csv.DictReader(f)}
            cmp_rows = []
            for r in rows:
                p = prev_rows.get(r["dataset"])
                if not p:
                    continue
                cmp_rows.append(
                    {
                        "dataset": r["dataset"],
                        "acc_delta": float(r["accuracy"]) - float(p["accuracy"]),
                        "fpr_delta": float(r["fpr"]) - float(p["fpr"]),
                        "bot_recall_delta": float(r["bot_recall"]) - float(p["bot_recall"]),
                    }
                )
            if cmp_rows:
                cmp_path = run_dir / "compare_with_prev.csv"
                with cmp_path.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["dataset", "acc_delta", "fpr_delta", "bot_recall_delta"])
                    w.writeheader()
                    w.writerows(cmp_rows)

        print(f"[run_dann_auto] completed version={run_dir.name}")
    finally:
        if args.cleanup_trials and trials_dir.is_dir():
            shutil.rmtree(trials_dir)
            print(f"[run_dann_auto] cleaned trials dir: {trials_dir}")

    print(f"[run_dann_auto] outputs={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
