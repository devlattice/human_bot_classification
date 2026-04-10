"""
Runtime forward for tabular SSL masked-AE embeddings (``ssl_masked_ae.npz``).

Must match ``workspace/ssl_data/ssl_embed/scripts/export_embeddings.py`` so
miner inference aligns with offline LGBM training on ``emb_*`` (± originals).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def load_ssl_masked_ae(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {
        "feature_cols": [str(x) for x in z["feature_cols"].tolist()],
        "mu": np.asarray(z["mu"], dtype=np.float64),
        "sigma": np.asarray(z["sigma"], dtype=np.float64),
        "coefs": [np.asarray(w, dtype=np.float64) for w in z["coefs"].tolist()],
        "intercepts": [np.asarray(b, dtype=np.float64) for b in z["intercepts"].tolist()],
        "embed_layer_index": int(np.asarray(z["embed_layer_index"]).reshape(-1)[0]),
    }


def _forward_to_layer(
    X: np.ndarray,
    coefs: list[np.ndarray],
    intercepts: list[np.ndarray],
    layer_idx: int,
) -> np.ndarray:
    H = X
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        H = H @ W + b
        if i <= layer_idx:
            H = _relu(H)
        if i == layer_idx:
            return H
    raise ValueError(f"Invalid embed layer index {layer_idx}")


def ssl_embedding_from_row(row: dict[str, float], art: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build normalized input, return (Z, X_raw) both shape (1, d).

    ``X_raw`` matches training concat: pre-norm feature values in ``feature_cols`` order.
    """
    feats: list[str] = art["feature_cols"]
    X = np.array([[float(row.get(c, 0.0)) for c in feats]], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = art["mu"]
    sigma = art["sigma"]
    Xn = (X - mu) / sigma
    Z = _forward_to_layer(Xn, art["coefs"], art["intercepts"], art["embed_layer_index"])
    return Z, X


def augment_row_with_ssl_embeddings(
    row: dict[str, float],
    art: dict[str, Any],
    model_features: list[str],
) -> dict[str, float]:
    """
    Fill ``emb_*`` from the encoder; other names are copied from ``row`` (concat = raw X in export script).
    """
    Z, _X_raw = ssl_embedding_from_row(row, art)
    zrow = Z.reshape(-1)
    out: dict[str, float] = {}
    for name in model_features:
        if name.startswith("emb_"):
            suffix = name[4:]
            try:
                idx = int(suffix)
            except ValueError:
                out[name] = 0.0
                continue
            if 0 <= idx < len(zrow):
                out[name] = float(zrow[idx])
            else:
                out[name] = 0.0
        else:
            out[name] = float(row.get(name, 0.0))
    return out
