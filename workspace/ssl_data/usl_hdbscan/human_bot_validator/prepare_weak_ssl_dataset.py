#!/usr/bin/env python3
"""
Step-by-step: build train/val parquets for weak cluster-based pseudo-labels + sample weights.

**Step 1 — Load** ``mixed_train_with_clusters.parquet`` (same row order as mix + HDBSCAN).

**Step 2 — Real labeled rows** ``mix_source`` in ``train_human`` / ``train_bot`` with non-null ``label``.

**Step 3 — Validation (real labels only, never pseudo):** either stratified split from
**labeled** mixed rows (``--val-fraction``), or copy an external parquet with
``--val-parquet`` (e.g. explorer ``train_v2_robust/val.parquet``) so **all** labeled
mixed rows stay in ``train.parquet`` — no internal holdout.

**Step 4 — Pseudo-labeled validator** (optional): ``mix_source == validator``, ``label`` is NA,
``cluster`` not -1, ``cluster_probability >= min-prob``. Map ``cluster`` → ``label`` via
``--cluster-human`` / ``--cluster-bot`` (defaults: 0→human, 1→bot).

**Step 5 — Weights** ``sample_weight``: 1.0 on real train rows; on pseudo rows
``pseudo_weight * cluster_probability`` (weak signal).

**Step 6 — Optional agreement** (``--agreement logistic``): fit logistic regression on the
**labeled train split** only; keep pseudo rows only where model prediction matches cluster label.

**Step 7 — Write** ``train.parquet`` (features + ``label`` + ``sample_weight``), ``val.parquet``
(features + ``label``), and ``ssl_prepare_summary.json``.

Train with e.g.::

  PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \\
    --data-dir <out-dir> \\
    --sample-weight-col sample_weight \\
    --out-dir workspace/model/artifacts/...

From repo root::

  PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/prepare_weak_ssl_dataset.py \\
    --input workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet \\
    --out-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/ssl_weak_step1 \\
    --val-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust/val.parquet \\
    --min-cluster-prob 0.7 --pseudo-weight 0.15 --seed 42

Canonical copy: this file. ``workspace/ssl_data/SSL/scripts/prepare_weak_ssl_dataset.py`` is a shim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]

META_COLS = frozenset(
    {
        "label",
        "mix_source",
        "cluster",
        "cluster_probability",
        "sample_weight",
    }
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def main() -> int:
    default_in = (
        REPO_ROOT
        / "workspace"
        / "ssl_data"
        / "usl_hdbscan"
        / "human_bot_validator"
        / "mixed_train_with_clusters.parquet"
    )
    default_out = (
        REPO_ROOT
        / "workspace"
        / "ssl_data"
        / "usl_hdbscan"
        / "human_bot_validator"
        / "ssl_weak_step1"
    )

    ap = argparse.ArgumentParser(
        description="Prepare train/val parquets: real labels + weak cluster pseudo-labels + weights."
    )
    ap.add_argument("--input", type=Path, default=default_in, help="mixed_train_with_clusters.parquet")
    ap.add_argument("--out-dir", type=Path, default=default_out, help="Writes train.parquet, val.parquet")
    ap.add_argument(
        "--val-parquet",
        type=Path,
        default=None,
        help="If set, use this labeled val (real features + label); all labeled mixed rows go to train. "
        "Must have same feature columns as --input (order follows mixed). Ignores --val-fraction.",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Stratified holdout from labeled mixed rows only when --val-parquet is omitted",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cluster-prob", type=float, default=0.7, help="Min cluster_probability for pseudo")
    ap.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.15,
        help="Scale for pseudo rows: weight = pseudo_weight * cluster_probability",
    )
    ap.add_argument(
        "--pseudo-fraction",
        type=float,
        default=1.0,
        help="Random fraction of pseudo rows to keep (1.0 = all passing filters)",
    )
    ap.add_argument("--cluster-human", type=int, default=0, help="Cluster id mapped to label 0 (human)")
    ap.add_argument("--cluster-bot", type=int, default=1, help="Cluster id mapped to label 1 (bot)")
    ap.add_argument(
        "--no-pseudo",
        action="store_true",
        help="Only real labeled rows in train (after split); no validator pseudo-labels",
    )
    ap.add_argument(
        "--agreement",
        choices=("none", "logistic"),
        default="none",
        help="If logistic: drop pseudo rows where a quick model disagrees with cluster→label",
    )
    args = ap.parse_args()

    path = args.input.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not path.is_file():
        print(f"[prepare_weak_ssl] error: missing input {path}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(int(args.seed))

    df = pd.read_parquet(path)
    for c in ("label", "mix_source", "cluster", "cluster_probability"):
        if c not in df.columns:
            print(f"[prepare_weak_ssl] error: column {c!r} missing", file=sys.stderr)
            return 1

    feat_cols = _feature_columns(df)
    if not feat_cols:
        print("[prepare_weak_ssl] error: no feature columns after excluding metadata", file=sys.stderr)
        return 1

    ch, cb = int(args.cluster_human), int(args.cluster_bot)
    if ch == cb:
        print("[prepare_weak_ssl] error: cluster-human and cluster-bot must differ", file=sys.stderr)
        return 1

    # --- Step 2: labeled train-source rows
    src = df["mix_source"].astype(str)
    is_train_src = src.isin(("train_human", "train_bot"))
    lab_ok = df["label"].notna()
    labeled_idx = df.index[is_train_src & lab_ok]
    labeled = df.loc[labeled_idx].copy()
    labeled["label"] = pd.to_numeric(labeled["label"], errors="coerce").astype("Int64")
    labeled = labeled[labeled["label"].notna() & labeled["label"].isin((0, 1))]
    y_lab = labeled["label"].astype(int).to_numpy()

    if len(labeled) < 10:
        print(f"[prepare_weak_ssl] error: too few labeled rows: {len(labeled)}", file=sys.stderr)
        return 1

    # --- Step 3: val = external file OR stratified holdout from labeled mixed only
    val_parquet_path: Path | None = None
    vf = float(args.val_fraction)

    if args.val_parquet is not None:
        val_parquet_path = Path(args.val_parquet).expanduser().resolve()
        if not val_parquet_path.is_file():
            print(f"[prepare_weak_ssl] error: --val-parquet not found {val_parquet_path}", file=sys.stderr)
            return 1
        val_ext = pd.read_parquet(val_parquet_path)
        if "label" not in val_ext.columns:
            print("[prepare_weak_ssl] error: external val missing `label`", file=sys.stderr)
            return 1
        ext_feats = [c for c in val_ext.columns if c != "label"]
        if set(ext_feats) != set(feat_cols):
            only_m = sorted(set(feat_cols) - set(ext_feats))
            only_v = sorted(set(ext_feats) - set(feat_cols))
            print(
                f"[prepare_weak_ssl] error: val feature columns differ from mixed "
                f"(only_in_mixed={only_m[:12]}{'...' if len(only_m) > 12 else ''} "
                f"only_in_val={only_v[:12]}{'...' if len(only_v) > 12 else ''})",
                file=sys.stderr,
            )
            return 1
        val_rows = val_ext[feat_cols + ["label"]].copy()
        val_rows["label"] = pd.to_numeric(val_rows["label"], errors="coerce")
        val_rows = val_rows[val_rows["label"].notna() & val_rows["label"].isin((0, 1))]
        val_rows["label"] = val_rows["label"].astype(int)
        val_rows = val_rows.reset_index(drop=True)
        fit_rows = labeled.reset_index(drop=True)
    else:
        if not (0.0 < vf < 0.5):
            print("[prepare_weak_ssl] error: val-fraction should be in (0, 0.5)", file=sys.stderr)
            return 1

        from sklearn.model_selection import train_test_split

        idx = np.arange(len(labeled))
        try:
            i_fit, i_val = train_test_split(
                idx,
                test_size=vf,
                random_state=int(args.seed),
                stratify=y_lab,
            )
        except ValueError as e:
            print(f"[prepare_weak_ssl] error: stratified split failed: {e}", file=sys.stderr)
            return 1

        fit_rows = labeled.iloc[i_fit].reset_index(drop=True)
        val_rows = labeled.iloc[i_val].reset_index(drop=True)

    # --- Step 4–5: pseudo from validator
    pseudo_parts: list[pd.DataFrame] = []
    n_pseudo_raw = 0
    n_pseudo_after = 0
    n_agreement_dropped = 0

    if not args.no_pseudo:
        is_val = src == "validator"
        unl = df["label"].isna()
        cl = pd.to_numeric(df["cluster"], errors="coerce")
        prob = pd.to_numeric(df["cluster_probability"], errors="coerce")
        mask = is_val & unl & cl.notna() & prob.notna()
        mask &= cl != -1
        mask &= prob >= float(args.min_cluster_prob)
        mask &= cl.isin((ch, cb))
        pseudo = df.loc[mask].copy()
        n_pseudo_raw = len(pseudo)

        if n_pseudo_raw > 0:
            pseudo["label"] = np.where(pseudo["cluster"].to_numpy() == ch, 0, 1).astype(np.int64)
            pseudo["sample_weight"] = float(args.pseudo_weight) * prob.loc[mask].to_numpy(dtype=float)

            pf = float(args.pseudo_fraction)
            if 0.0 < pf < 1.0:
                keep = rng.random(len(pseudo)) < pf
                pseudo = pseudo.iloc[np.where(keep)[0]].copy()

            # --- Step 6: optional agreement
            if args.agreement == "logistic" and len(pseudo) > 0 and len(fit_rows) >= 20:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                X_fit = fit_rows[feat_cols].to_numpy(dtype=float)
                y_fit = fit_rows["label"].astype(int).to_numpy()
                X_p = pseudo[feat_cols].to_numpy(dtype=float)

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=200,
                                class_weight="balanced",
                                random_state=int(args.seed),
                            ),
                        ),
                    ]
                )
                pipe.fit(X_fit, y_fit)
                pred = pipe.predict(X_p)
                agree = pred == pseudo["label"].to_numpy()
                n_agreement_dropped = int(np.sum(~agree))
                pseudo = pseudo.loc[agree].reset_index(drop=True)

            n_pseudo_after = len(pseudo)
            if len(pseudo) > 0:
                pseudo_parts.append(pseudo)

    fit_rows = fit_rows.copy()
    fit_rows["sample_weight"] = 1.0

    if pseudo_parts:
        train_df = pd.concat([fit_rows, pd.concat(pseudo_parts, ignore_index=True)], ignore_index=True)
    else:
        train_df = fit_rows

    train_df = train_df.reset_index(drop=True)
    val_df = val_rows[feat_cols + ["label"]].reset_index(drop=True)
    train_out = train_df[feat_cols + ["label", "sample_weight"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)

    summary: dict[str, Any] = {
        "input": str(path),
        "out_dir": str(out_dir),
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "val_mode": "external_parquet" if val_parquet_path is not None else "stratified_from_labeled_mixed",
        "val_parquet": str(val_parquet_path) if val_parquet_path is not None else None,
        "val_fraction": vf if val_parquet_path is None else None,
        "min_cluster_prob": float(args.min_cluster_prob),
        "pseudo_weight_scale": float(args.pseudo_weight),
        "pseudo_fraction": float(args.pseudo_fraction),
        "no_pseudo": bool(args.no_pseudo),
        "agreement": args.agreement,
        "cluster_human": ch,
        "cluster_bot": cb,
        "counts": {
            "labeled_total": int(len(labeled)),
            "train_fit_real": int(len(fit_rows)),
            "val_holdout_real": int(len(val_df)),
            "pseudo_candidates": int(n_pseudo_raw),
            "pseudo_after_filters": int(n_pseudo_after),
            "pseudo_dropped_agreement": int(n_agreement_dropped),
            "train_total_rows": int(len(train_out)),
        },
    }
    (out_dir / "ssl_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["counts"], indent=2))
    print(f"[prepare_weak_ssl] wrote {out_dir / 'train.parquet'} and val.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
