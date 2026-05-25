#!/usr/bin/env python3
"""
Pretrain a simple masked autoencoder (MLPRegressor) on tabular features.

Implementation note:
- Uses sklearn MLPRegressor to reconstruct original features from masked inputs.
- Exports a lightweight artifact (`ssl_masked_ae.npz`) with:
  - feature columns
  - normalization stats
  - MLP weights
  - embedding-layer index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def _read_feature_list(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _load_matrix(parquet_path: Path, features: list[str]) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet_path}: missing features ({len(missing)}): {missing[:10]}")
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def _read_weight_map(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--mask-weight-json must be a JSON object: {feature: weight}")
    out: dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _weights_for_features(
    feature_names: list[str],
    weight_map: dict[str, float],
) -> np.ndarray:
    w = np.array([max(0.0, float(weight_map.get(f, 0.0))) for f in feature_names], dtype=np.float64)
    if float(w.sum()) <= 0.0:
        w = np.ones(len(feature_names), dtype=np.float64)
    w /= float(w.sum())
    return w


def _mask_random(X: np.ndarray, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    M = rng.random(X.shape) < float(mask_ratio)
    Xm = X.copy()
    Xm[M] = 0.0
    return Xm


def _mask_weighted(
    X: np.ndarray,
    mask_ratio: float,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n, d = X.shape
    k = max(1, int(round(float(mask_ratio) * d)))
    Xm = X.copy()
    cols = np.arange(d, dtype=np.int64)
    for i in range(n):
        pick = rng.choice(cols, size=k, replace=False, p=weights)
        Xm[i, pick] = 0.0
    return Xm


def _mask_inputs(
    X: np.ndarray,
    *,
    mask_ratio: float,
    mask_mode: str,
    mixed_alpha: float,
    weights: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    mode = str(mask_mode).strip().lower()
    if mode == "random":
        return _mask_random(X, mask_ratio, rng)
    if mode == "weighted":
        if weights is None:
            raise ValueError("mask_mode=weighted requires --mask-weight-json")
        return _mask_weighted(X, mask_ratio, weights, rng)
    if mode == "mixed":
        if weights is None:
            raise ValueError("mask_mode=mixed requires --mask-weight-json")
        alpha = float(np.clip(mixed_alpha, 0.0, 1.0))
        n = X.shape[0]
        use_weighted = rng.random(n) < alpha
        Xm = X.copy()
        if np.any(~use_weighted):
            Xm[~use_weighted] = _mask_random(X[~use_weighted], mask_ratio, rng)
        if np.any(use_weighted):
            Xm[use_weighted] = _mask_weighted(X[use_weighted], mask_ratio, weights, rng)
        return Xm
    raise ValueError(f"unknown --mask-mode {mask_mode!r} (choose random|weighted|mixed)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pretrain masked tabular autoencoder (sklearn MLP).")
    ap.add_argument("--pool-parquet", type=Path, required=True, help="Pooled SSL parquet.")
    ap.add_argument("--feature-cols-file", type=Path, required=True, help="Ordered feature list file.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Artifacts output directory.")
    ap.add_argument("--mask-ratio", type=float, default=0.30)
    ap.add_argument(
        "--mask-mode",
        choices=("random", "weighted", "mixed"),
        default="random",
        help="Masking scheme over feature dimensions.",
    )
    ap.add_argument(
        "--mask-weight-json",
        type=Path,
        default=None,
        help="JSON mapping {feature_name: weight}; used by weighted/mixed masking.",
    )
    ap.add_argument(
        "--mask-mixed-alpha",
        type=float,
        default=0.30,
        help="For mask_mode=mixed: fraction of rows using weighted masking.",
    )
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    features = _read_feature_list(args.feature_cols_file.expanduser().resolve())
    if not features:
        raise SystemExit("feature list is empty")
    pool_path = args.pool_parquet.expanduser().resolve()
    if not pool_path.is_file():
        raise SystemExit(f"missing --pool-parquet {pool_path}")

    X = _load_matrix(pool_path, features)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma
    rng = np.random.default_rng(int(args.seed))
    weight_map: dict[str, float] = {}
    if args.mask_weight_json is not None:
        weight_map = _read_weight_map(args.mask_weight_json.expanduser().resolve())
    weights: np.ndarray | None = None
    if weight_map:
        weights = _weights_for_features(features, weight_map)
    Xm = _mask_inputs(
        Xn,
        mask_ratio=float(args.mask_ratio),
        mask_mode=str(args.mask_mode),
        mixed_alpha=float(args.mask_mixed_alpha),
        weights=weights,
        rng=rng,
    )

    hidden = (int(args.hidden_dim), int(args.embed_dim), int(args.hidden_dim))
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(1024, max(64, len(Xm) // 40)),
        learning_rate_init=1e-3,
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        verbose=True,
    )
    model.fit(Xm, Xn)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "ssl_masked_ae.npz"
    np.savez_compressed(
        npz_path,
        feature_cols=np.array(features, dtype=object),
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64),
        coefs=np.array([w.astype(np.float64) for w in model.coefs_], dtype=object),
        intercepts=np.array([b.astype(np.float64) for b in model.intercepts_], dtype=object),
        embed_layer_index=np.array([1], dtype=np.int64),  # h1 -> h2 (embedding)
    )

    meta = {
        "pool_parquet": str(pool_path),
        "n_rows": int(len(X)),
        "n_features": int(len(features)),
        "feature_cols_file": str(args.feature_cols_file.expanduser().resolve()),
        "mask_ratio": float(args.mask_ratio),
        "mask_mode": str(args.mask_mode),
        "mask_weight_json": (
            str(args.mask_weight_json.expanduser().resolve()) if args.mask_weight_json is not None else None
        ),
        "mask_mixed_alpha": float(args.mask_mixed_alpha),
        "hidden_dim": int(args.hidden_dim),
        "embed_dim": int(args.embed_dim),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "artifact_npz": str(npz_path),
        "loss": float(model.loss_) if hasattr(model, "loss_") else None,
    }
    (out_dir / "ssl_masked_ae.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[pretrain_masked_ae] wrote {npz_path} rows={len(X)} features={len(features)} "
        f"embed_dim={args.embed_dim}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

