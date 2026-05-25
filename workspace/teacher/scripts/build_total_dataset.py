#!/usr/bin/env python3
"""
Merge **full** gold-labeled train with pseudo-labeled validator rows for LightGBM.

No random split and no dropping gold rows: ``--labeled-train`` is concatenated with all
pseudo rows (validator hands with teacher labels). Real validation comes only from your
gold ``--labeled-val`` file (default: public ``val.parquet``); that file is copied with
``sample_weight`` added (default 1.0) so schema matches ``train.parquet``—LightGBM still
uses labels on val for early stopping; weights on val are excluded from features via
``--sample-weight-col``.

Drops teacher-only columns from pseudo: ``p_dann``, ``p_lgbm``, ``p_teacher``,
``teacher_agreement``, ``confidence_band``. Maps ``y_hat`` → ``label`` and
``pseudo_weight`` → ``sample_weight``. Gold **train** rows use ``--gold-weight``.

Writes ``train.parquet`` and ``val.parquet`` under ``--out-dir``.

Optional ``--run-qc`` runs ``train_qc`` after the manifest is written (thresholds fit on
gold train rows only; outputs ``qc/<run_id>/`` and adds ``train_qc`` to ``manifest.json``).

Use with::

  python workspace/model/scripts/lgbm_2.py --data-dir ... --sample-weight-col sample_weight

Run from repo root::

  PYTHONPATH=. python workspace/teacher/scripts/build_total_dataset.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

# Removed from pseudo before aligning to gold feature columns (y_hat / pseudo_weight handled separately).
TEACHER_DROP = frozenset(
    {
        "p_dann",
        "p_dann_a",
        "p_dann_b",
        "p_lgbm",
        "p_teacher",
        "teacher_agreement",
        "confidence_band",
    }
)
SAMPLE_WEIGHT_COL = "sample_weight"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge labeled train + pseudo-labeled validator for LGBM.")
    p.add_argument(
        "--labeled-train",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "data"
        / "public"
        / "train.parquet",
        help="Gold train.parquet (must include `label`).",
    )
    p.add_argument(
        "--pseudo-labeled-validator",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "teacher"
        / "artifacts"
        / "pseudo_teacher_validator"
        / "pseudo_labeled_validator.parquet",
        help="Output from build_teacher_pseudo_labels.py.",
    )
    p.add_argument(
        "--labeled-val",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "data"
        / "public"
        / "val.parquet",
        help="Gold-only validation parquet (same feature schema as train). Gets sample_weight only (default 1.0).",
    )
    p.add_argument(
        "--val-weight",
        type=float,
        default=1.0,
        help="Constant sample_weight for every gold val row (schema parity with train).",
    )
    p.add_argument(
        "--no-val",
        action="store_true",
        help="Do not write val.parquet.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "workspace" / "teacher" / "artifacts" / "total_dataset",
        help="Directory for train.parquet (+ val.parquet unless --no-val).",
    )
    p.add_argument(
        "--gold-weight",
        type=float,
        default=1.0,
        help="sample_weight for gold-labeled **train** rows (not pseudo).",
    )
    p.add_argument(
        "--run-qc",
        action="store_true",
        help="After writing manifest, run train_qc (writes qc/<run_id>/ and updates manifest).",
    )
    p.add_argument(
        "--qc-fail-on-gate",
        action="store_true",
        help="With --run-qc: exit non-zero if train_qc gate fails.",
    )
    p.add_argument(
        "--qc-max-gold-hard-frac",
        type=float,
        default=0.02,
        help="train_qc gate: max allowed hard-tier fraction on gold train rows.",
    )
    p.add_argument(
        "--qc-max-val-hard-frac",
        type=float,
        default=0.05,
        help="train_qc gate: max allowed hard-tier fraction on gold val.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_path = args.labeled_train.expanduser().resolve()
    pseudo_path = args.pseudo_labeled_validator.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.is_file():
        raise SystemExit(f"Missing --labeled-train: {train_path}")
    if not pseudo_path.is_file():
        raise SystemExit(f"Missing --pseudo-labeled-validator: {pseudo_path}")

    gold = pd.read_parquet(train_path)
    if "label" not in gold.columns:
        raise SystemExit("labeled train must contain `label`")
    feature_cols = [c for c in gold.columns if c != "label"]

    pseudo = pd.read_parquet(pseudo_path)
    if "y_hat" not in pseudo.columns or "pseudo_weight" not in pseudo.columns:
        raise SystemExit("Pseudo parquet must contain `y_hat` and `pseudo_weight`.")

    y_hat = pd.to_numeric(pseudo["y_hat"], errors="coerce").fillna(0).astype(np.int64)
    pseudo_w = pd.to_numeric(pseudo["pseudo_weight"], errors="coerce").fillna(0.0)

    drop_list = [c for c in TEACHER_DROP if c in pseudo.columns]
    if "label" in pseudo.columns:
        drop_list.append("label")
    pseudo_f = pseudo.drop(columns=drop_list + ["y_hat", "pseudo_weight"], errors="ignore")

    missing = [c for c in feature_cols if c not in pseudo_f.columns]
    if missing:
        raise SystemExit(f"Pseudo parquet missing features (compare to labeled train): {missing[:20]}")

    pseudo_rows = pseudo_f.loc[:, feature_cols].copy()
    pseudo_rows.insert(0, "label", y_hat)
    pseudo_rows[SAMPLE_WEIGHT_COL] = pseudo_w.to_numpy(dtype=np.float64)

    gold_rows = gold.copy()
    if SAMPLE_WEIGHT_COL in gold_rows.columns:
        gold_rows = gold_rows.drop(columns=[SAMPLE_WEIGHT_COL])
    gold_rows[SAMPLE_WEIGHT_COL] = float(args.gold_weight)

    # Same column order as gold train: label, features..., sample_weight
    gold_cols = list(gold_rows.columns)
    pseudo_rows = pseudo_rows.loc[:, gold_cols]

    merged = pd.concat([gold_rows, pseudo_rows], axis=0, ignore_index=True)

    train_out = out_dir / "train.parquet"
    merged.to_parquet(train_out, index=False)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_by": "build_total_dataset.py",
        "labeled_train": str(train_path),
        "pseudo_labeled_validator": str(pseudo_path),
        "out_train": str(train_out),
        "n_gold": int(len(gold_rows)),
        "n_pseudo": int(len(pseudo_rows)),
        "n_total_train": int(len(merged)),
        "n_feature_cols": len(feature_cols),
        "sample_weight_col": SAMPLE_WEIGHT_COL,
        "gold_weight": float(args.gold_weight),
        "val_weight": float(args.val_weight),
    }

    if not args.no_val:
        val_path = args.labeled_val.expanduser().resolve()
        if val_path.is_file():
            val_df = pd.read_parquet(val_path)
            if "label" not in val_df.columns:
                raise SystemExit("--labeled-val must contain `label`")
            if SAMPLE_WEIGHT_COL in val_df.columns:
                val_df = val_df.drop(columns=[SAMPLE_WEIGHT_COL])
            v_feats = [c for c in val_df.columns if c != "label"]
            if set(v_feats) != set(feature_cols):
                raise SystemExit(
                    "Val features must match labeled-train features. "
                    f"symmetric_diff={sorted(set(feature_cols) ^ set(v_feats))[:20]}"
                )
            # Same column order as train parquet: label, features (train order), sample_weight
            val_out_df = val_df.loc[:, ["label", *feature_cols]].copy()
            val_out_df[SAMPLE_WEIGHT_COL] = float(args.val_weight)
            val_out_df = val_out_df.loc[:, gold_cols]
            vo = out_dir / "val.parquet"
            val_out_df.to_parquet(vo, index=False)
            manifest["labeled_val"] = str(val_path)
            manifest["out_val"] = str(vo)
            manifest["n_val"] = int(len(val_out_df))
        else:
            print(
                f"[warn] No val.parquet written (missing file: {val_path}). lgbm_2.py needs train+val.",
                file=sys.stderr,
            )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if args.run_qc:
        sys.path.insert(0, str(REPO_ROOT / "workspace" / "teacher"))
        from train_qc.pipeline import run_train_qc_on_bundle

        qc_out = run_train_qc_on_bundle(
            out_dir,
            run_id=None,
            max_gold_hard_frac=float(args.qc_max_gold_hard_frac),
            max_val_hard_frac=float(args.qc_max_val_hard_frac),
        )
        print("[train_qc]", json.dumps(qc_out, indent=2), file=sys.stderr)
        if args.qc_fail_on_gate and qc_out.get("gate") == "fail":
            raise SystemExit("train_qc gate failed (--qc-fail-on-gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
