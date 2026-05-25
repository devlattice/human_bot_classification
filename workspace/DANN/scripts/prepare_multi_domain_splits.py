#!/usr/bin/env python3
"""
Create reproducible tune/test splits for multiple labeled domain parquets.

This is a convenience wrapper around split_labeled_parquet.py behavior:
  - writes <prefix>_tune.parquet and <prefix>_test.parquet under per-domain split dirs
  - prints a ready-to-run run_dann_auto.py command using those files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _split_one(
    *,
    parquet: Path,
    out_dir: Path,
    label_col: str,
    tune_frac: float,
    seed: int,
    prefix: str,
) -> tuple[Path, Path]:
    df = pd.read_parquet(parquet)
    if label_col not in df.columns:
        raise SystemExit(f"Missing label column {label_col!r} in {parquet}")
    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any():
        raise SystemExit(f"Non-numeric label values in {parquet} for column {label_col!r}")
    y01 = (y > 0.5).astype(int)

    idx = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).to_numpy()
    idx0 = [i for i in idx if y01.iloc[i] == 0]
    idx1 = [i for i in idx if y01.iloc[i] == 1]

    n0_t = int(round(len(idx0) * tune_frac))
    n1_t = int(round(len(idx1) * tune_frac))
    tune_idx = idx0[:n0_t] + idx1[:n1_t]
    test_idx = idx0[n0_t:] + idx1[n1_t:]

    out_dir.mkdir(parents=True, exist_ok=True)
    tune_path = out_dir / f"{prefix}_tune.parquet"
    test_path = out_dir / f"{prefix}_test.parquet"

    df_tune = df.iloc[tune_idx].sample(frac=1.0, random_state=seed + 11).reset_index(drop=True)
    df_test = df.iloc[test_idx].sample(frac=1.0, random_state=seed + 29).reset_index(drop=True)
    df_tune.to_parquet(tune_path, index=False)
    df_test.to_parquet(test_path, index=False)
    return tune_path, test_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare multi-domain tune/test splits and print run command.")
    p.add_argument(
        "--domain-parquet",
        action="append",
        type=Path,
        default=[],
        help="Labeled domain parquet to split (repeatable).",
    )
    p.add_argument(
        "--out-base-dir",
        type=Path,
        required=True,
        help="Base output dir for split folders (one subdir per prefix).",
    )
    p.add_argument("--label-col", default="label")
    p.add_argument("--tune-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional prefix per --domain-parquet (same order).",
    )
    p.add_argument("--train-parquet", type=Path, required=True)
    p.add_argument("--val-parquet", type=Path, required=True)
    p.add_argument("--validator-parquet", type=Path, required=True)
    p.add_argument("--grid-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--target-human-fpr", type=float, default=0.05)
    p.add_argument("--threshold-grid-size", type=int, default=401)
    p.add_argument("--threshold-tie-ref", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.domain_parquet:
        raise SystemExit("Provide at least one --domain-parquet")
    if not (0.0 < args.tune_frac < 1.0):
        raise SystemExit("--tune-frac must be in (0,1)")
    if args.prefix and len(args.prefix) != len(args.domain_parquet):
        raise SystemExit("--prefix count must match --domain-parquet count when provided")

    domain_paths = [Path(p).expanduser().resolve() for p in args.domain_parquet]
    for p in domain_paths:
        if not p.is_file():
            raise SystemExit(f"Missing domain parquet: {p}")

    prefixes: list[str] = []
    for i, p in enumerate(domain_paths):
        if args.prefix:
            prefixes.append(args.prefix[i])
        else:
            prefixes.append(p.stem)

    out_base = Path(args.out_base_dir).expanduser().resolve()
    tune_paths: list[Path] = []
    test_paths: list[Path] = []
    for i, (p, pref) in enumerate(zip(domain_paths, prefixes)):
        out_dir = out_base / pref
        tune_p, test_p = _split_one(
            parquet=p,
            out_dir=out_dir,
            label_col=args.label_col,
            tune_frac=float(args.tune_frac),
            seed=int(args.seed) + i * 101,
            prefix=pref,
        )
        tune_paths.append(tune_p)
        test_paths.append(test_p)
        print(f"[prepare_multi_domain_splits] {p}")
        print(f"  tune: {tune_p}")
        print(f"  test: {test_p}")

    cmd = [
        "python workspace/DANN/scripts/run_dann_auto.py",
        f"--train-parquet {Path(args.train_parquet).expanduser().resolve()}",
        f"--val-parquet {Path(args.val_parquet).expanduser().resolve()}",
        f"--validator-parquet {Path(args.validator_parquet).expanduser().resolve()}",
        f"--grid-json {Path(args.grid_json).expanduser().resolve()}",
        "--val-selection-metric multi_objective_generalization",
        f"--target-human-fpr {float(args.target_human_fpr)}",
        f"--threshold-grid-size {int(args.threshold_grid_size)}",
        f"--threshold-tie-ref {float(args.threshold_tie_ref)}",
        f"--device {args.device}",
        "--cleanup-trials",
    ]
    cmd.extend([f"--tune-parquet {p}" for p in tune_paths])
    cmd.extend([f"--test-parquet {p}" for p in test_paths])
    print("\n[prepare_multi_domain_splits] run command:")
    print(" \\\n  ".join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

