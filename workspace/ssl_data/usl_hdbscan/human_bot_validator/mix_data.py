#!/usr/bin/env python3
"""
Mix **labeled** train rows (equal human / bot counts) with **validator** robust train rows, then shuffle.

- ``--real-source``: parquet with ``label`` in ``{0, 1}`` (0 = human, 1 = bot).
- ``--validator-source``: parquet in the same feature layout (robusted); all rows tagged ``validator``.
- Output: single shuffled parquet under ``--output-dir``.
- ``--summary``: print ``mix_source`` / ``label`` breakdown (validator is usually all ``label`` NA).
- ``--copy-manifest``: copy ``build_manifest.json`` into ``--output-dir`` so ``cluster_hdbscan.py`` finds
  ``hdbscan_feature_columns`` next to ``mixed_train.parquet`` (optional ``--manifest-src``).

Run from repo root::

  PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \\
    --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \\
    --summary --copy-manifest

  PYTHONPATH=. python .../mix_data.py \\
    --real-source workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet \\
    --validator-source workspace/ssl_data/raw_data/miner_1/validator_request_robusted.parquet \\
    --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \\
    --n-per-class 3000 --summary --copy-manifest

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet \
  --validator-source workspace/ssl_data/raw_data/miner_1/validator_request_robusted.parquet \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \
  --n-per-class 3000 \
  --summary \
  --copy-manifest
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    default_real = (
        REPO_ROOT
        / "workspace"
        / "dataset"
        / "robusted_dataset"
        / "train"
        / "system_human_bot"
        / "train.parquet"
    )
    default_val = REPO_ROOT / "workspace" / "ssl_data" / "robusted_data" / "train.parquet"
    default_out = REPO_ROOT / "workspace" / "ssl_data" / "usl_hdbscan" / "human_bot_validator"
    default_manifest = REPO_ROOT / "workspace" / "ssl_data" / "usl_hdbscan" / "data" / "build_manifest.json"

    ap = argparse.ArgumentParser(description="Mix sampled labeled train + validator parquet, shuffle.")
    ap.add_argument(
        "--real-source",
        type=Path,
        default=default_real,
        help=f"Labeled train parquet (default: {default_real})",
    )
    ap.add_argument(
        "--validator-source",
        type=Path,
        default=default_val,
        help=f"Validator robust train parquet (default: {default_val})",
    )
    ap.add_argument(
        "--n-per-class",
        type=int,
        default=3000,
        help="Sample this many rows with label=0 and this many with label=1 (default: 3000).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Directory for output parquet (default: {default_out})",
    )
    ap.add_argument(
        "--output-name",
        default="mixed_train.parquet",
        help="Output filename (default: mixed_train.parquet).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print mix_source and label breakdown after writing (validator rows usually have label NA).",
    )
    ap.add_argument(
        "--copy-manifest",
        action="store_true",
        help="Copy build_manifest.json into --output-dir so cluster_hdbscan can auto-pick feature columns.",
    )
    ap.add_argument(
        "--manifest-src",
        type=Path,
        default=None,
        help="Source JSON for --copy-manifest (default: workspace/ssl_data/usl_hdbscan/data/build_manifest.json).",
    )
    args = ap.parse_args()

    n = int(args.n_per_class)
    if n < 1:
        print("[mix_data] error: --n-per-class must be >= 1", file=sys.stderr)
        return 1

    real_path = args.real_source.expanduser().resolve()
    val_path = args.validator_source.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not real_path.is_file():
        print(f"[mix_data] error: missing --real-source {real_path}", file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"[mix_data] error: missing --validator-source {val_path}", file=sys.stderr)
        return 1

    df_real = pd.read_parquet(real_path)
    df_val = pd.read_parquet(val_path)

    if "label" not in df_real.columns:
        print("[mix_data] error: real-source must have a label column.", file=sys.stderr)
        return 1

    if "label" not in df_val.columns:
        df_val = df_val.copy()
        df_val["label"] = pd.NA

    h = df_real[df_real["label"] == 0]
    b = df_real[df_real["label"] == 1]
    if len(h) < n or len(b) < n:
        print(
            f"[mix_data] error: need at least {n} rows per class; have label=0: {len(h)}, label=1: {len(b)}",
            file=sys.stderr,
        )
        return 1

    rng = int(args.seed)
    samp_h = h.sample(n=n, random_state=rng)
    samp_b = b.sample(n=n, random_state=rng + 1)

    ref_cols = list(df_real.columns)
    missing_val = [c for c in ref_cols if c not in df_val.columns]
    if missing_val:
        print(
            f"[mix_data] error: validator-source missing columns present in real-source: {missing_val[:24]}",
            file=sys.stderr,
        )
        return 1

    df_val_aligned = df_val[ref_cols].copy()

    samp_h = samp_h.copy()
    samp_b = samp_b.copy()
    samp_h["mix_source"] = "train_human"
    samp_b["mix_source"] = "train_bot"
    df_val_aligned = df_val_aligned.copy()
    df_val_aligned["mix_source"] = "validator"

    mixed = pd.concat([samp_h, samp_b, df_val_aligned], axis=0, ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    mixed.to_parquet(out_path, index=False)

    print(
        f"[mix_data] wrote {out_path} rows={len(mixed)} "
        f"(train_human={n} train_bot={n} validator={len(df_val_aligned)}) cols={len(mixed.columns)}"
    )

    if args.copy_manifest:
        msrc = args.manifest_src.expanduser().resolve() if args.manifest_src is not None else default_manifest
        mdst = out_dir / "build_manifest.json"
        if not msrc.is_file():
            print(f"[mix_data] warning: --copy-manifest skipped (missing {msrc})", file=sys.stderr)
        else:
            shutil.copy2(msrc, mdst)
            print(f"[mix_data] copied manifest → {mdst}")

    if args.summary:
        print("\n[mix_data] --summary mix_source counts:")
        print(mixed["mix_source"].value_counts().sort_index().to_string())
        print("\n[mix_data] --summary label by mix_source (non-null / NA):")
        for src in sorted(mixed["mix_source"].unique()):
            sub = mixed.loc[mixed["mix_source"] == src, "label"]
            nn = sub.notna().sum()
            na = sub.isna().sum()
            if nn and pd.api.types.is_numeric_dtype(sub):
                vc = pd.to_numeric(sub.dropna(), errors="coerce").astype("Int64").value_counts().sort_index()
                extra = f" label values: {vc.to_dict()}"
            elif nn:
                extra = f" label value_counts: {sub.dropna().value_counts().to_dict()}"
            else:
                extra = ""
            print(f"  {src}: non_null={int(nn)} na={int(na)}{extra}")
        print(
            "\n[mix_data] note: validator rows have no human/bot label here unless your "
            "--validator-source parquet already includes label 0/1."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
