#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise SystemExit("PyTorch is required. Install torch first.") from e
    return torch, nn


def _build_encoder(nn, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, embed_dim),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export embeddings from trained DL adapter artifact.")
    p.add_argument("--artifact", type=Path, required=True, help="Path to dl_adapter.pt")
    p.add_argument("--in-parquet", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--concat-original-features", action="store_true")
    p.add_argument("--passthrough-col", action="append", default=[])
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    torch, nn = _require_torch()
    artifact_path = args.artifact.expanduser().resolve()
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which rejects
    # non-tensor metadata in our local trusted artifact payload.
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    feature_cols: list[str] = [str(x) for x in payload["feature_cols"]]
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    hidden_dim = int(payload["hidden_dim"])
    embed_dim = int(payload["embed_dim"])
    dropout = float(payload["dropout"])

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)

    encoder = _build_encoder(nn, len(feature_cols), hidden_dim, embed_dim, dropout)
    state_dict = payload["state_dict"]
    enc_state = {k[len("encoder.") :]: v for k, v in state_dict.items() if str(k).startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.to(device)
    encoder.eval()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for in_path in args.in_parquet:
        inp = in_path.expanduser().resolve()
        df = pd.read_parquet(inp)
        miss = [c for c in feature_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{inp}: missing {len(miss)} feature columns, e.g. {miss[:3]}")
        x = df[feature_cols].to_numpy(dtype=np.float32)
        x = (x - mean) / std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        with torch.no_grad():
            z = encoder(torch.from_numpy(x).to(device)).detach().cpu().numpy().astype(np.float32)
        z_cols = [f"adp_{i:03d}" for i in range(z.shape[1])]
        out_df = pd.DataFrame(z, columns=z_cols)
        if args.concat_original_features:
            out_df = pd.concat([out_df, df[feature_cols].reset_index(drop=True)], axis=1)
        for col in args.passthrough_col:
            if col in df.columns:
                out_df[col] = df[col].values
        if "label" in df.columns and "label" not in out_df.columns:
            out_df["label"] = df["label"].values
        out_path = out_dir / inp.name
        out_df.to_parquet(out_path, index=False)
        written.append(str(out_path))
        print(f"[export] {inp} -> {out_path} rows={len(out_df)}", flush=True)

    report = {
        "artifact": str(artifact_path),
        "in_inputs": [str(Path(p).expanduser().resolve()) for p in args.in_parquet],
        "written": written,
        "concat_original_features": bool(args.concat_original_features),
        "passthrough_cols": list(args.passthrough_col),
        "adapter_embed_dim": embed_dim,
        "adapter_feature_count": len(feature_cols),
    }
    (out_dir / "export_adapter_embeddings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

