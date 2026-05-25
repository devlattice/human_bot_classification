#!/usr/bin/env python3
"""
Run a **small** Cartesian grid of `train_dann.py` hyperparameters and record
`best_score` / `best_val_acc` from each trial's ``*.meta.json`` (see ``train_dann.py``).

Why not a huge grid
--------------------
Each trial trains a full model. Prefer **coarse** grids + **fewer epochs** for screening,
then a longer run on the best few settings. Random search is another option for high
dimensional spaces.

Example grid JSON (lists are combined with ``itertools.product``); see
``hparam_grid_example.json`` (24 trials on a typical GPU screen)::

  {
    "lambda_max": [0.05, 0.1, 0.2, 0.35],
    "lambda_gamma": [6.0, 10.0, 14.0],
    "lr": [0.0003, 0.001]
  }

Fixed keys like ``source-npz`` / ``target-npz`` / ``epochs`` come from CLI.

Example::

  python workspace/DANN/scripts/hparam_grid_search.py \\
    --source-npz workspace/DANN/artifacts/source_train.npz \\
    --target-npz workspace/DANN/artifacts/target_train.npz \\
    --grid-json workspace/DANN/scripts/hparam_grid_example.json \\
    --epochs 40 \\
    --out-csv workspace/DANN/artifacts/hparam_results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPTS = Path(__file__).resolve().parent
_TRAIN = _SCRIPTS / "train_dann.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search wrapper for train_dann.py")
    p.add_argument("--source-npz", type=Path, required=True)
    p.add_argument(
        "--source-val-npz",
        type=Path,
        default=None,
        help="Optional held-out source val npz passed to train_dann.py",
    )
    p.add_argument(
        "--extra-val-npz",
        action="append",
        type=Path,
        default=[],
        help="Optional extra labeled validation npz passed to train_dann.py (repeatable).",
    )
    p.add_argument("--target-npz", type=Path, required=True)
    p.add_argument("--grid-json", type=Path, required=True, help="JSON object: param -> list of values")
    p.add_argument("--epochs", type=int, default=40, help="Epochs per trial (use small for screening)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device passed to train_dann.py (e.g. cpu, cuda).",
    )
    p.add_argument("--trials-dir", type=Path, default=None, help="Where to write trial_*/ checkpoints")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
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
        help="Forwarded to train_dann.py (checkpoint + threshold selection; default matches deployment objective).",
    )
    p.add_argument(
        "--target-human-fpr",
        type=float,
        default=0.05,
        help="Forwarded to train_dann.py when using bot_recall_at_human_fpr_cap.",
    )
    p.add_argument("--threshold-grid-size", type=int, default=401)
    p.add_argument("--threshold-tie-ref", type=float, default=0.5)
    p.add_argument(
        "--hybrid-nuisance-seat",
        action="store_true",
        help="Forward to train_dann.py (requires seat_bucket in labeled npz).",
    )
    p.add_argument("--hybrid-nuisance-weight", type=float, default=0.5)
    p.add_argument("--n-seat-buckets", type=int, default=9)
    return p.parse_args()


def _cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    rows = []
    for combo in itertools.product(*vals):
        rows.append(dict(zip(keys, combo)))
    return rows


def main() -> None:
    args = parse_args()
    grid_path = args.grid_json.expanduser().resolve()
    raw = json.loads(grid_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("--grid-json must be a non-empty JSON object of param -> list")

    combos = _expand_grid(raw)
    trials_root = args.trials_dir
    if trials_root is None:
        trials_root = _SCRIPTS.parent / "artifacts" / "hparam_trials"
    trials_root = Path(trials_root).expanduser().resolve()
    trials_root.mkdir(parents=True, exist_ok=True)

    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial",
        "best_score",
        "best_val_acc",
        "val_selection_metric",
        "selected_threshold",
        "val_human_fpr_at_selected_threshold",
        "val_bot_recall_at_selected_threshold",
        "val_threshold_feasible",
        "domain_confusion_loss_source_val_vs_target",
        "domain_confusion_acc_source_val_vs_target",
        "n_extra_val_domains",
        "ckpt",
        *sorted(raw.keys()),
    ]

    rows_out: List[Dict[str, Any]] = []

    for i, h in enumerate(combos):
        trial_dir = trials_root / f"trial_{i:04d}"
        ckpt = trial_dir / "dann.pt"
        meta_path = ckpt.with_suffix(".meta.json")
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            sys.executable,
            str(_TRAIN),
            "--source-npz",
            str(args.source_npz.expanduser().resolve()),
            "--target-npz",
            str(args.target_npz.expanduser().resolve()),
            "--out",
            str(ckpt),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.source_val_npz is not None:
            cmd.extend(["--source-val-npz", str(args.source_val_npz.expanduser().resolve())])
        for p in args.extra_val_npz:
            cmd.extend(["--extra-val-npz", str(p.expanduser().resolve())])
        if args.device:
            cmd.extend(["--device", args.device])
        cmd.extend(
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
            cmd.append("--hybrid-nuisance-seat")
            cmd.extend(
                [
                    "--hybrid-nuisance-weight",
                    str(args.hybrid_nuisance_weight),
                    "--n-seat-buckets",
                    str(args.n_seat_buckets),
                ]
            )
        for k, v in h.items():
            cmd.extend([_cli_flag(k), str(v)])

        print(f"[hparam_grid_search] trial {i+1}/{len(combos)}: {h}", flush=True)
        if args.dry_run:
            print(" ", " ".join(cmd), flush=True)
            rows_out.append(
                {
                    "trial": i,
                    "best_score": "",
                    "best_val_acc": "",
                    "val_selection_metric": str(args.val_selection_metric),
                    "selected_threshold": "",
                    "val_human_fpr_at_selected_threshold": "",
                    "val_bot_recall_at_selected_threshold": "",
                    "val_threshold_feasible": "",
                    "domain_confusion_loss_source_val_vs_target": "",
                    "domain_confusion_acc_source_val_vs_target": "",
                    "n_extra_val_domains": len(args.extra_val_npz),
                    "ckpt": str(ckpt),
                    **h,
                }
            )
            continue

        r = subprocess.run(cmd, cwd=str(_SCRIPTS.parent.parent.parent))
        if r.returncode != 0:
            print(f"[hparam_grid_search] trial {i} FAILED (exit {r.returncode})", flush=True)
            rows_out.append(
                {
                    "trial": i,
                    "best_score": float("nan"),
                    "best_val_acc": float("nan"),
                    "val_selection_metric": str(args.val_selection_metric),
                    "selected_threshold": float("nan"),
                    "val_human_fpr_at_selected_threshold": float("nan"),
                    "val_bot_recall_at_selected_threshold": float("nan"),
                    "val_threshold_feasible": "",
                    "domain_confusion_loss_source_val_vs_target": float("nan"),
                    "domain_confusion_acc_source_val_vs_target": float("nan"),
                    "n_extra_val_domains": len(args.extra_val_npz),
                    "ckpt": str(ckpt),
                    **h,
                }
            )
            continue

        meta_sel = str(args.val_selection_metric)
        br = float("nan")
        hf = float("nan")
        st = float("nan")
        feas_s = ""
        if not meta_path.is_file():
            print(f"[hparam_grid_search] missing {meta_path}", flush=True)
            best = float("nan")
            score = float("nan")
            dloss = float("nan")
            dacc = float("nan")
            nexd = len(args.extra_val_npz)
        else:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_sel = str(meta.get("val_selection_metric", args.val_selection_metric))
            best = float(meta.get("best_val_acc", float("nan")))
            score = float(meta.get("best_score", meta.get("best_val_acc", float("nan"))))
            st = float(meta.get("selected_threshold", float("nan")))
            hf = float(meta.get("val_human_fpr_at_selected_threshold", float("nan")))
            br = float(meta.get("val_bot_recall_at_selected_threshold", float("nan")))
            feas_s = str(int(bool(meta.get("val_threshold_feasible", True))))
            dloss = float(meta.get("domain_confusion_loss_source_val_vs_target", float("nan")))
            dacc = float(meta.get("domain_confusion_acc_source_val_vs_target", float("nan")))
            nexd = int(meta.get("n_extra_val_domains", len(args.extra_val_npz)))

        row = {
            "trial": i,
            "best_score": score,
            "best_val_acc": best,
            "val_selection_metric": meta_sel,
            "selected_threshold": st,
            "val_human_fpr_at_selected_threshold": hf,
            "val_bot_recall_at_selected_threshold": br,
            "val_threshold_feasible": feas_s,
            "domain_confusion_loss_source_val_vs_target": dloss,
            "domain_confusion_acc_source_val_vs_target": dacc,
            "n_extra_val_domains": nexd,
            "ckpt": str(ckpt),
            **h,
        }
        rows_out.append(row)
        if meta_sel in (
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ):
            print(
                f"[hparam_grid_search] best_score={score:.6f}  val_bot_recall@thr={br:.6f}  "
                f"val_human_fpr={hf:.6f}  thr={st:.4f}  feasible={feas_s}  "
                f"domain_conf_loss={dloss:.6f}  domain_acc={dacc:.6f}",
                flush=True,
            )
        else:
            print(
                f"[hparam_grid_search] best_score={score:.6f}  best_val_acc@0.5={best:.6f}",
                flush=True,
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"[hparam_grid_search] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
