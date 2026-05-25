#!/usr/bin/env python3
"""
Export tabular embeddings from `ssl_masked_ae.npz` artifact.

By default writes (concat mode):
- emb_000 .. emb_{k-1}, then original feature columns (same order as training)
- label column if present in input parquet
- optional extra columns via ``--passthrough-col`` (e.g. ``sample_weight`` for weighted LGBM)

``concat`` does not mean "copy the whole input row": the script **rebuilds** the table from the
artifact's ``feature_cols`` + embeddings, so metadata columns are dropped unless passed through.

Use --embedding-only to write emb_* columns only (no original features).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _load_artifact(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "feature_cols": [str(x) for x in z["feature_cols"].tolist()],
        "mu": np.asarray(z["mu"], dtype=np.float64),
        "sigma": np.asarray(z["sigma"], dtype=np.float64),
        "coefs": [np.asarray(w, dtype=np.float64) for w in z["coefs"].tolist()],
        "intercepts": [np.asarray(b, dtype=np.float64) for b in z["intercepts"].tolist()],
        "embed_layer_index": int(np.asarray(z["embed_layer_index"]).reshape(-1)[0]),
    }


def _forward_to_layer(X: np.ndarray, coefs: list[np.ndarray], intercepts: list[np.ndarray], layer_idx: int) -> np.ndarray:
    H = X
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        H = H @ W + b
        if i <= layer_idx:
            H = _relu(H)
        if i == layer_idx:
            return H
    raise ValueError(f"Invalid embed layer index {layer_idx}")


def _embed_df(
    df: pd.DataFrame,
    art: dict,
    concat_original: bool,
    passthrough_cols: list[str] | None = None,
) -> pd.DataFrame:
    features = art["feature_cols"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"missing features ({len(missing)}): {missing[:10]}")
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    Xn = (X - art["mu"]) / art["sigma"]
    Z = _forward_to_layer(Xn, art["coefs"], art["intercepts"], art["embed_layer_index"])

    out = pd.DataFrame({f"emb_{i:03d}": Z[:, i] for i in range(Z.shape[1])})
    if concat_original:
        for j, c in enumerate(features):
            out[c] = X[:, j]
    if "label" in df.columns:
        out.insert(0, "label", pd.to_numeric(df["label"], errors="coerce"))

    for col in passthrough_cols or []:
        if col not in df.columns:
            raise ValueError(f"--passthrough-col {col!r}: column missing from input parquet")
        if col in out.columns:
            raise ValueError(f"--passthrough-col {col!r}: conflicts with embedding output column")
        out[col] = df[col]

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export embeddings parquet from masked-AE artifact.")
    ap.add_argument("--artifact", type=Path, required=True, help="ssl_masked_ae.npz")
    ap.add_argument("--in-parquet", action="append", default=[], help="Repeatable input parquet.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    ap.add_argument(
        "--embedding-only",
        action="store_true",
        help="Write only emb_* (and label if present). Default: concat original features after emb_*.",
    )
    ap.add_argument(
        "--concat-original-features",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--passthrough-col",
        action="append",
        default=[],
        metavar="NAME",
        help="Repeatable. Copy this column from input to output (e.g. sample_weight). "
        "Must not clash with emb_* or artifact feature names.",
    )
    args = ap.parse_args()

    art_path = args.artifact.expanduser().resolve()
    if not art_path.is_file():
        raise SystemExit(f"missing --artifact {art_path}")
    in_paths = [Path(p).expanduser().resolve() for p in args.in_parquet]
    if not in_paths:
        raise SystemExit("Provide at least one --in-parquet")
    for p in in_paths:
        if not p.is_file():
            raise SystemExit(f"missing input parquet: {p}")

    art = _load_artifact(art_path)
    passthrough = [str(c).strip() for c in (args.passthrough_col or []) if str(c).strip()]
    for col in passthrough:
        if col in art["feature_cols"] or col.startswith("emb_"):
            raise SystemExit(f"--passthrough-col {col!r}: reserved (feature or emb_* name)")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for p in in_paths:
        df = pd.read_parquet(p)
        out_df = _embed_df(
            df,
            art,
            concat_original=not bool(args.embedding_only),
            passthrough_cols=passthrough,
        )
        out_path = out_dir / p.name
        out_df.to_parquet(out_path, index=False)
        written.append(str(out_path))
        print(f"[export_embeddings] wrote {out_path} rows={len(out_df)} cols={len(out_df.columns)}", file=sys.stderr)

    meta = {
        "artifact": str(art_path),
        "n_inputs": int(len(in_paths)),
        "written": written,
        "concat_original_features": not bool(args.embedding_only),
        "passthrough_cols": passthrough,
    }
    (out_dir / "export_embeddings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

