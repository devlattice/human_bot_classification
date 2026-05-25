#!/usr/bin/env python3
"""
Mix **labeled** train rows (equal human / bot counts) with **validator** robust train rows, then shuffle.

- ``--real-source``: parquet with ``label`` in ``{0, 1}`` (0 = human, 1 = bot).
- ``--validator-source``: parquet in the same feature layout (robusted); all rows tagged ``validator``.
- Output: single shuffled parquet under ``--output-dir``.
- ``--summary``: print ``mix_source`` / ``label`` breakdown (validator is usually all ``label`` NA).
- ``--copy-manifest``: copy ``build_manifest.json`` into ``--output-dir`` so ``cluster_hdbscan.py`` finds
  ``hdbscan_feature_columns`` next to ``mixed_train.parquet`` (optional ``--manifest-src``).
- By default writes summary PNGs under ``--plot-dir`` (``<output-dir>/plots``). Pass ``--no-plot`` to skip.
- ``--intersect-features``: keep only columns present in **both** parquets (plus ``label`` from real if
  validator lacks it). Use when train was featurized with a **superset** of validator columns.
- ``--extra-source-1`` / ``--extra-source-2``: optional extra parquet(s) to append as additional mix sources.
  Use ``--extra-source-*-rate`` as a fraction of sampled labeled rows (``2 * n_per_class``).

Run from repo root::

  PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \\
    --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \\
    --summary --copy-manifest

  PYTHONPATH=. python .../mix_data.py \\
    --real-source workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet \\
    --validator-source workspace/ssl_data/raw_data/miner_1/validator_request_robusted.parquet \\
    --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \\
    --n-per-class 3000 --summary --copy-manifest

  # Use all available balanced train rows (e.g. 4000+4000 when val has 6638 rows):
  PYTHONPATH=. python .../mix_data.py ... --n-per-class auto --summary --copy-manifest

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet \
  --validator-source workspace/ssl_data/raw_data/feature_2/validator.parquet \
  --extra-source-1 workspace/preprocess/statistical_test/explorer/feature_2/data/irc/irc_train.parquet \
  --extra-source-1-rate 0.2 \
  --extra-source-2 /path/to/another_extra.parquet \
  --extra-source-2-rate 0.3 \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/data \
  --n-per-class auto \
  --intersect-features \
  --summary
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_mix_plots(mixed: pd.DataFrame, plot_dir: Path, stem: str) -> list[Path]:
    """Bar charts: mix_source counts and train label 0/1. Returns written paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[mix_data] error: pip install matplotlib (or use --no-plot)", file=sys.stderr)
        raise

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    vc = mixed["mix_source"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(vc) + 1)))
    vc.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_xlabel("row count")
    ax.set_ylabel("mix_source")
    ax.set_title("Mixed dataset: rows per mix_source")
    fig.tight_layout()
    p1 = plot_dir / f"{stem}_mix_source_counts.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(p1)

    if "label" in mixed.columns:
        sub = mixed[mixed["mix_source"].isin(("train_human", "train_bot"))].copy()
        sub["_l"] = pd.to_numeric(sub["label"], errors="coerce")
        sub = sub[sub["_l"].notna()]
        if len(sub) > 0:
            cts = sub["_l"].astype(int).value_counts().sort_index()
            labels = [f"label={i} ({'human' if i == 0 else 'bot'})" for i in cts.index]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(labels, cts.values, color=["#4c72b0", "#dd8452"][: len(cts)])
            ax2.set_ylabel("count")
            ax2.set_title("Labeled train rows in mix (human + bot sample)")
            fig2.tight_layout()
            p2 = plot_dir / f"{stem}_train_label_counts.png"
            fig2.savefig(p2, dpi=120)
            plt.close(fig2)
            written.append(p2)

    return written


def _align_extra_to_ref(df_extra: pd.DataFrame, ref_cols: list[str], src_name: str) -> pd.DataFrame:
    """Align extra source to ref columns; fill missing cols with NA and drop extras."""
    out = df_extra.copy()
    missing = [c for c in ref_cols if c not in out.columns]
    if missing:
        preview = missing[:24]
        more = f" … (+{len(missing) - len(preview)} more)" if len(missing) > len(preview) else ""
        print(
            f"[mix_data] warning: {src_name} missing {len(missing)} ref columns; filling NA: {preview}{more}",
            file=sys.stderr,
        )
        for c in missing:
            out[c] = pd.NA
    return out.loc[:, ref_cols].copy()


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
        type=str,
        default="3000",
        help="Sample this many rows per class (label=0 and label=1). Use integer, or 'auto' / 'max' "
        "to use min(count_humans, count_bots) in --real-source (default: 3000).",
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
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for summary PNGs (default: <output-dir>/plots).",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib summary plots.",
    )
    ap.add_argument(
        "--intersect-features",
        action="store_true",
        help="Use columns in the intersection of both parquets (preserve real-source column order). "
        "Drops real-only feature columns with a stderr warning. Fails if no shared features remain.",
    )
    ap.add_argument("--extra-source-1", type=Path, default=None, help="Optional parquet to append as extra_source_1.")
    ap.add_argument(
        "--extra-source-1-rate",
        type=float,
        default=0.0,
        help="Sample rate for extra_source_1 relative to (2 * n_per_class), e.g. 0.2.",
    )
    ap.add_argument("--extra-source-2", type=Path, default=None, help="Optional parquet to append as extra_source_2.")
    ap.add_argument(
        "--extra-source-2-rate",
        type=float,
        default=0.0,
        help="Sample rate for extra_source_2 relative to (2 * n_per_class), e.g. 0.3.",
    )
    args = ap.parse_args()

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
    max_per_class = min(len(h), len(b))
    n_raw = (args.n_per_class or "").strip().lower()
    if n_raw in ("auto", "max"):
        n = max_per_class
        print(f"[mix_data] --n-per-class {n_raw!r} → using {n} per class (min of label=0/1 counts).", file=sys.stderr)
    else:
        try:
            n = int(args.n_per_class, 10)
        except ValueError:
            print(
                f"[mix_data] error: --n-per-class must be an integer or 'auto'/'max', got {args.n_per_class!r}",
                file=sys.stderr,
            )
            return 1
    if n < 1:
        print("[mix_data] error: --n-per-class must be >= 1", file=sys.stderr)
        return 1
    if len(h) < n or len(b) < n:
        print(
            f"[mix_data] error: need at least {n} rows per class; have label=0: {len(h)}, label=1: {len(b)}. "
            f"Use --n-per-class {max_per_class} or --n-per-class auto",
            file=sys.stderr,
        )
        return 1

    rng = int(args.seed)
    samp_h = h.sample(n=n, random_state=rng)
    samp_b = b.sample(n=n, random_state=rng + 1)

    real_order = list(df_real.columns)
    val_set = set(df_val.columns)
    if args.intersect_features:
        ref_cols = [c for c in real_order if c == "label" or c in val_set]
        dropped = [c for c in real_order if c != "label" and c not in val_set]
        if dropped:
            nd = len(dropped)
            preview = dropped[:32]
            more = f" … (+{nd - len(preview)} more)" if nd > len(preview) else ""
            print(
                f"[mix_data] --intersect-features: dropped {nd} real-only columns not in validator: "
                f"{preview}{more}",
                file=sys.stderr,
            )
        feat_only = [c for c in ref_cols if c != "label"]
        if not feat_only:
            print(
                "[mix_data] error: --intersect-features left no shared feature columns (only label?).",
                file=sys.stderr,
            )
            return 1
    else:
        ref_cols = real_order
        missing_val = [c for c in ref_cols if c not in df_val.columns]
        if missing_val:
            print(
                f"[mix_data] error: validator-source missing columns present in real-source: {missing_val[:24]}",
                file=sys.stderr,
            )
            print(
                "[mix_data] hint: rebuild validator with the same feature pipeline, or retry with "
                "--intersect-features to use only columns present in both parquets.",
                file=sys.stderr,
            )
            return 1

    samp_h = samp_h.loc[:, ref_cols]
    samp_b = samp_b.loc[:, ref_cols]

    df_val_aligned = df_val[ref_cols].copy()

    samp_h = samp_h.copy()
    samp_b = samp_b.copy()
    samp_h["mix_source"] = "train_human"
    samp_b["mix_source"] = "train_bot"
    df_val_aligned = df_val_aligned.copy()
    df_val_aligned["mix_source"] = "validator"

    parts = [samp_h, samp_b, df_val_aligned]
    base_labeled_rows = 2 * n

    for i, (src_arg, rate_arg) in enumerate(
        (
            (args.extra_source_1, args.extra_source_1_rate),
            (args.extra_source_2, args.extra_source_2_rate),
        ),
        start=1,
    ):
        if src_arg is None:
            continue
        src_path = src_arg.expanduser().resolve()
        if not src_path.is_file():
            print(f"[mix_data] error: missing --extra-source-{i} {src_path}", file=sys.stderr)
            return 1
        rate = float(rate_arg)
        if rate < 0:
            print(f"[mix_data] error: --extra-source-{i}-rate must be >= 0", file=sys.stderr)
            return 1
        take_n = int(round(base_labeled_rows * rate))
        if take_n <= 0:
            print(f"[mix_data] --extra-source-{i}: rate={rate} -> sample size 0; skipped", file=sys.stderr)
            continue
        df_extra = pd.read_parquet(src_path)
        if "label" not in df_extra.columns:
            df_extra = df_extra.copy()
            df_extra["label"] = pd.NA
        df_extra = _align_extra_to_ref(df_extra, ref_cols, f"extra_source_{i}")
        if len(df_extra) == 0:
            print(f"[mix_data] --extra-source-{i}: empty parquet; skipped", file=sys.stderr)
            continue
        take_n = min(take_n, len(df_extra))
        samp_extra = df_extra.sample(n=take_n, random_state=rng + 100 + i).copy()
        samp_extra["mix_source"] = f"extra_source_{i}"
        parts.append(samp_extra)
        print(
            f"[mix_data] --extra-source-{i}: rate={rate} sampled={take_n} (available={len(df_extra)})",
            file=sys.stderr,
        )

    mixed = pd.concat(parts, axis=0, ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    mixed.to_parquet(out_path, index=False)

    print(
        f"[mix_data] wrote {out_path} rows={len(mixed)} "
        f"(train_human={n} train_bot={n} validator={len(df_val_aligned)}) cols={len(mixed.columns)} "
        f"sources={sorted(mixed['mix_source'].unique().tolist())}"
    )

    if args.copy_manifest:
        msrc = args.manifest_src.expanduser().resolve() if args.manifest_src is not None else default_manifest
        mdst = out_dir / "build_manifest.json"
        if not msrc.is_file():
            print(f"[mix_data] warning: --copy-manifest skipped (missing {msrc})", file=sys.stderr)
        else:
            shutil.copy2(msrc, mdst)
            print(f"[mix_data] copied manifest → {mdst}")
            if args.intersect_features:
                print(
                    "[mix_data] warning: manifest hdbscan_feature_columns may include dropped columns; "
                    "edit build_manifest.json or pass --manifest to cluster_hdbscan to match intersected features.",
                    file=sys.stderr,
                )

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

    if not args.no_plot:
        plot_dir = args.plot_dir.expanduser().resolve() if args.plot_dir is not None else (out_dir / "plots")
        stem = Path(args.output_name).stem
        try:
            paths = _write_mix_plots(mixed, plot_dir, stem)
            for p in paths:
                print(f"[mix_data] plot {p}")
        except ImportError:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
