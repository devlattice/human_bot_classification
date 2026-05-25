#!/usr/bin/env python3
"""Run inference with a DANN checkpoint (task logits / probabilities)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from model import DANN


def load_checkpoint(path: Path, device: torch.device) -> tuple[DANN, Dict[str, Any], np.ndarray, np.ndarray]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    meta = ckpt.get("meta") or {}
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    std = np.asarray(ckpt["scaler_std"], dtype=np.float64)
    in_dim = int(meta["in_dim"])
    feat_dim = int(meta.get("feat_dim", 64))
    hidden_dim = int(meta.get("hidden_dim", 128))
    dropout = float(meta.get("dropout", 0.1))
    use_nuis = bool(meta.get("use_nuisance_seat", False))
    n_seat = int(meta.get("n_seat_buckets", 9))
    model = DANN(
        in_dim=in_dim,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        grl_lambda=1.0,
        use_nuisance_seat=use_nuis,
        n_seat_buckets=n_seat,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta, mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DANN inference on feature matrix.")
    p.add_argument("--ckpt", type=Path, required=True, help="train_dann.py checkpoint .pt")
    p.add_argument("--npz", type=Path, default=None, help="npz with key X")
    p.add_argument("--out-npz", type=Path, default=None, help="Write probs to npz (key p_bot)")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu). Use --device cpu to force CPU.",
    )
    return p.parse_args()


def _resolve_device(explicit: str | None) -> torch.device:
    if explicit is not None and explicit.strip() != "":
        if explicit == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    if args.npz is None:
        raise SystemExit("--npz required")

    z = np.load(args.npz, allow_pickle=False)
    if "X" not in z.files:
        raise ValueError(f"expected key 'X', got {z.files}")
    X = np.asarray(z["X"], dtype=np.float32)

    model, _meta, mean, std = load_checkpoint(args.ckpt, device)
    Xn = apply_scaler(X, mean, std)
    xt = torch.from_numpy(Xn).to(device)
    logits = model(xt).task_logits
    probs = torch.sigmoid(logits).cpu().numpy()

    for i, p in enumerate(probs[:10]):
        print(f"{i}: p_bot={p:.6f}")
    if len(probs) > 10:
        print(f"... ({len(probs)} rows total)")

    if args.out_npz is not None:
        out = Path(args.out_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, p_bot=probs.astype(np.float32))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
