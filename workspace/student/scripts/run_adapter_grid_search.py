#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


GRID_KEYS = (
    "embed_dim",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "lambda_domain_max",
    "lambda_domain_gamma",
    "domain_selection_weight",
    "domain_eval_target_rows",
)


def _load_grid(path: Path) -> dict[str, list[Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    out: dict[str, list[Any]] = {}
    for k in GRID_KEYS:
        vals = payload.get(k)
        if vals is None:
            raise ValueError(f"{path}: missing key {k!r}")
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"{path}: key {k!r} must be non-empty list")
        out[k] = vals
    return out


def _cmd_for_trial(args: argparse.Namespace, trial_dir: Path, hp: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "workspace" / "student" / "scripts" / "train_dl_adapter.py"),
        "--source-train",
        str(args.source_train),
        "--source-val",
        str(args.source_val),
        "--target-unlabeled",
        str(args.target_unlabeled),
        "--feature-cols-file",
        str(args.feature_cols_file),
        "--out-dir",
        str(trial_dir),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--device",
        str(args.device),
        "--embed-dim",
        str(hp["embed_dim"]),
        "--hidden-dim",
        str(hp["hidden_dim"]),
        "--dropout",
        str(hp["dropout"]),
        "--lr",
        str(hp["lr"]),
        "--weight-decay",
        str(hp["weight_decay"]),
        "--lambda-domain-max",
        str(hp["lambda_domain_max"]),
        "--lambda-domain-gamma",
        str(hp["lambda_domain_gamma"]),
        "--domain-selection-weight",
        str(hp["domain_selection_weight"]),
        "--domain-eval-target-rows",
        str(hp["domain_eval_target_rows"]),
        "--target-human-fpr",
        str(args.target_human_fpr),
        "--threshold-grid-size",
        str(args.threshold_grid_size),
        "--threshold-tie-ref",
        str(args.threshold_tie_ref),
    ]
    cmd.extend(["--val-selection-metric", str(args.val_selection_metric)])
    for p in args.extra_val_labeled:
        cmd.extend(["--extra-val-labeled", str(Path(p).expanduser().resolve())])
    return cmd


def _trial_score(
    metrics_path: Path, target_human_fpr: float
) -> tuple[tuple[float, ...], float, float, float, float, str]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric = str(payload.get("params", {}).get("val_selection_metric", "val_score_plus_domain_confusion"))
    best_score = payload.get("best_score")
    best_rank_raw = payload.get("best_rank")
    selection = payload.get("best_selection_score")
    domain_conf = payload.get("best_domain_confusion_score")
    val = payload.get("val_metrics_at_selected_threshold", {})
    bot_recall = float(val.get("recall", 0.0))
    human_fpr = float(val.get("human_fpr", 1.0))
    legacy_score = float(bot_recall - 0.25 * max(0.0, human_fpr - float(target_human_fpr)))
    score = float(best_score) if best_score is not None else (float(selection) if selection is not None else legacy_score)
    domain_confusion = float(domain_conf) if domain_conf is not None else float("nan")
    if isinstance(best_rank_raw, list) and best_rank_raw:
        sort_key = tuple(float(v) for v in best_rank_raw)
    elif metric == "val_score_plus_domain_confusion":
        sort_key = (float(score),)
    else:
        feas = 1.0 if human_fpr <= float(target_human_fpr) + 1e-12 else 0.0
        sort_key = (feas, float(score), -float(human_fpr))
    return sort_key, score, bot_recall, human_fpr, domain_confusion, metric


def main() -> None:
    p = argparse.ArgumentParser(description="Grid search for student DL adapter using hparam_grid_adapter.json.")
    p.add_argument("--source-train", type=Path, required=True)
    p.add_argument("--source-val", type=Path, required=True)
    p.add_argument("--target-unlabeled", type=Path, required=True)
    p.add_argument("--feature-cols-file", type=Path, required=True)
    p.add_argument(
        "--grid-json",
        type=Path,
        default=REPO_ROOT / "workspace" / "student" / "scripts" / "hparam_grid_adapter.json",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--target-human-fpr", type=float, default=0.05)
    p.add_argument("--threshold-grid-size", type=int, default=1001)
    p.add_argument("--threshold-tie-ref", type=float, default=0.5)
    p.add_argument(
        "--val-selection-metric",
        type=str,
        default="val_score_plus_domain_confusion",
        choices=(
            "val_score_plus_domain_confusion",
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ),
    )
    p.add_argument(
        "--extra-val-labeled",
        action="append",
        type=Path,
        default=[],
        help="Optional extra labeled validation parquet(s), repeatable.",
    )
    p.add_argument("--max-trials", type=int, default=0, help="0 means run full cartesian product.")
    args = p.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = _load_grid(args.grid_json.expanduser().resolve())

    combos = list(itertools.product(*(grid[k] for k in GRID_KEYS)))
    if args.max_trials and args.max_trials > 0:
        combos = combos[: int(args.max_trials)]

    summary_rows: list[dict[str, Any]] = []
    for i, vals in enumerate(combos):
        hp = {k: vals[j] for j, k in enumerate(GRID_KEYS)}
        trial_dir = out_dir / f"trial_{i:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd = _cmd_for_trial(args, trial_dir, hp)
        print(f"[grid] trial {i+1}/{len(combos)} -> {trial_dir.name} hp={hp}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        row: dict[str, Any] = {"trial": i, "status": int(proc.returncode), **hp}
        mpath = trial_dir / "adapter_metrics.json"
        if proc.returncode == 0 and mpath.is_file():
            rank_key, score, br, hf, dcs, sel_metric = _trial_score(
                mpath, target_human_fpr=float(args.target_human_fpr)
            )
            row.update(
                {
                    "val_selection_metric": sel_metric,
                    "rank_key": list(rank_key),
                    "selection_score": float(score),
                    "val_bot_recall": float(br),
                    "val_human_fpr": float(hf),
                    "best_domain_confusion_score": float(dcs),
                    "artifact": str((trial_dir / "dl_adapter.pt").resolve()),
                }
            )
        summary_rows.append(row)

    # Rank successful trials by lexicographic rank_key (larger is better).
    ranked = sorted(
        [r for r in summary_rows if r.get("status") == 0 and "selection_score" in r],
        key=lambda r: tuple(float(v) for v in r.get("rank_key", [])),
        reverse=True,
    )
    report = {
        "grid_json": str(args.grid_json.expanduser().resolve()),
        "total_trials": len(combos),
        "successful_trials": len(ranked),
        "best_trial": ranked[0] if ranked else None,
        "trials": summary_rows,
    }
    (out_dir / "grid_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ranked:
        print(json.dumps({"best_trial": ranked[0], "summary": str((out_dir / "grid_summary.json").resolve())}, indent=2))
    else:
        print(json.dumps({"message": "No successful trials", "summary": str((out_dir / "grid_summary.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()

