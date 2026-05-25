#!/usr/bin/env python3
"""Retrain student LGBM on pseudo-labeled parquet using existing adapter + best hparams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace.model.scripts.lgbm import _threshold_sweep, evaluate
from workspace.model.scripts.lgbm_2 import train_lgbm_b_v2


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


def _load_optuna_best(optuna_summary: Path) -> dict[str, Any]:
    payload = json.loads(optuna_summary.read_text(encoding="utf-8"))
    best = payload.get("best_params")
    if not isinstance(best, dict):
        raise SystemExit(f"{optuna_summary}: missing best_params")
    return payload


def _embed_df(df: pd.DataFrame, *, artifact: dict[str, Any], device: str, concat_original_features: bool) -> pd.DataFrame:
    torch, nn = _require_torch()
    feature_cols: list[str] = [str(x) for x in artifact["feature_cols"]]
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing adapter feature columns in input: {miss[:5]} (total {len(miss)})")

    mean = np.asarray(artifact["mean"], dtype=np.float32)
    std = np.asarray(artifact["std"], dtype=np.float32)
    hidden_dim = int(artifact["hidden_dim"])
    embed_dim = int(artifact["embed_dim"])
    dropout = float(artifact["dropout"])

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    dev_t = torch.device(dev)

    encoder = _build_encoder(nn, len(feature_cols), hidden_dim, embed_dim, dropout)
    state_dict = artifact["state_dict"]
    enc_state = {k[len("encoder.") :]: v for k, v in state_dict.items() if str(k).startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.to(dev_t)
    encoder.eval()

    x = df[feature_cols].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    with torch.no_grad():
        z = encoder(torch.from_numpy(x).to(dev_t)).detach().cpu().numpy().astype(np.float32)
    z_cols = [f"adp_{i:03d}" for i in range(z.shape[1])]
    out_df = pd.DataFrame(z, columns=z_cols)
    if concat_original_features:
        out_df = pd.concat([out_df, df[feature_cols].reset_index(drop=True)], axis=1)
    if "label" in df.columns:
        out_df["label"] = df["label"].values
    return out_df


def _eval_holdouts(
    model,
    holdout_parquets: list[Path],
    *,
    artifact: dict[str, Any],
    device: str,
    concat_original_features: bool,
    selected_threshold: float,
    feature_cols: list[str],
    emb_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fprs: list[float] = []
    for hp in holdout_parquets:
        src = hp.expanduser().resolve()
        df = pd.read_parquet(src)
        if "label" not in df.columns:
            raise SystemExit(f"{src}: holdout parquet must contain `label`")
        emb = _embed_df(
            df,
            artifact=artifact,
            device=device,
            concat_original_features=concat_original_features,
        )
        out_emb = emb_dir / f"holdout_{src.stem}.parquet"
        emb.to_parquet(out_emb, index=False)

        miss = [c for c in feature_cols if c not in emb.columns]
        if miss:
            raise SystemExit(f"{src}: embedded holdout missing feature columns: {miss[:5]}")
        X = emb[feature_cols]
        y = emb["label"].to_numpy()
        m = evaluate(model, X, y, threshold=selected_threshold)
        hfpr = float(m["human_fpr"])
        if np.isfinite(hfpr):
            fprs.append(hfpr)
        rows.append(
            {
                "path": str(src),
                "n": int(len(y)),
                "embedded_parquet": str(out_emb),
                "human_fpr": hfpr,
                "bot_recall": float(m["recall"]),
                "accuracy": float(m["accuracy"]),
                "roc_auc": float(m.get("roc_auc", float("nan"))),
            }
        )
    return {
        "per_holdout": rows,
        "max_human_fpr": max(fprs) if fprs else float("nan"),
        "mean_human_fpr": float(np.mean(fprs)) if fprs else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pseudo-dir", type=Path, required=True, help="Directory with train.parquet and val.parquet.")
    p.add_argument("--adapter-artifact", type=Path, required=True, help="Path to adapter/dl_adapter.pt.")
    p.add_argument("--optuna-summary", type=Path, required=True, help="Path to lgbm_optuna/optuna_summary.json.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output root dir for embeddings and retrained model.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for adapter embedding export.")
    p.add_argument("--lgbm-device", choices=("cpu", "gpu"), default="cpu", help="LightGBM training device.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concat-original-features", action="store_true", help="Concatenate adapter embeddings with original features.")
    p.add_argument(
        "--holdout-parquet",
        action="append",
        default=[],
        help="Optional holdout parquet(s) for post-train evaluation (repeatable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    train_path = pseudo_dir / "train.parquet"
    val_path = pseudo_dir / "val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit(f"{pseudo_dir}: expected train.parquet and val.parquet")

    adapter_path = args.adapter_artifact.expanduser().resolve()
    optuna_path = args.optuna_summary.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    emb_dir = out_dir / "embeddings"
    lgbm_out = out_dir / "lgbm_retrain"
    emb_dir.mkdir(parents=True, exist_ok=True)
    lgbm_out.mkdir(parents=True, exist_ok=True)

    torch, _ = _require_torch()
    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    optuna_payload = _load_optuna_best(optuna_path)
    best_params = dict(optuna_payload["best_params"])
    fpr_cap = float(optuna_payload.get("fpr_cap", 0.05))

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise SystemExit("Both train.parquet and val.parquet must contain `label`.")

    train_emb = _embed_df(
        train_df,
        artifact=adapter_payload,
        device=args.device,
        concat_original_features=bool(args.concat_original_features),
    )
    val_emb = _embed_df(
        val_df,
        artifact=adapter_payload,
        device=args.device,
        concat_original_features=bool(args.concat_original_features),
    )
    train_emb_path = emb_dir / "train.parquet"
    val_emb_path = emb_dir / "val.parquet"
    train_emb.to_parquet(train_emb_path, index=False)
    val_emb.to_parquet(val_emb_path, index=False)

    feature_cols = [c for c in train_emb.columns if c != "label"]
    X_train = train_emb[feature_cols]
    y_train = train_emb["label"].to_numpy()
    X_val = val_emb[feature_cols]
    y_val = val_emb["label"].to_numpy()

    model = train_lgbm_b_v2(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=int(args.seed),
        device=args.lgbm_device,
        log_every=50,
        sample_weight=None,
        n_estimators=int(best_params["n_estimators"]),
        learning_rate=float(best_params["learning_rate"]),
        num_leaves=int(best_params["num_leaves"]),
        max_depth=int(best_params["max_depth"]),
        min_child_samples=int(best_params["min_child_samples"]),
        subsample=float(best_params["subsample"]),
        colsample_bytree=float(best_params["colsample_bytree"]),
        reg_alpha=float(best_params["reg_alpha"]),
        reg_lambda=float(best_params["reg_lambda"]),
        min_gain_to_split=float(best_params["min_gain_to_split"]),
        early_stopping_rounds=int(best_params["early_stopping_rounds"]),
    )

    y_val_score = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
    sweep = _threshold_sweep(
        y_true=y_val,
        y_score=y_val_score,
        target_human_fpr=fpr_cap,
        grid_size=1001,
        threshold_tie_ref=0.5,
    )
    selected_threshold = float(sweep["selected_threshold"])
    train_metrics = evaluate(model, X_train, y_train, threshold=selected_threshold)
    val_metrics = evaluate(model, X_val, y_val, threshold=selected_threshold)
    holdouts = [Path(p).expanduser().resolve() for p in args.holdout_parquet]

    summary = {
        "pseudo_dir": str(pseudo_dir),
        "adapter_artifact": str(adapter_path),
        "optuna_summary": str(optuna_path),
        "best_params_reused": best_params,
        "fpr_cap": fpr_cap,
        "selected_threshold": selected_threshold,
        "train_metrics_at_selected_threshold": train_metrics,
        "val_metrics_at_selected_threshold": val_metrics,
        "n_features": int(len(feature_cols)),
        "concat_original_features": bool(args.concat_original_features),
        "embeddings_train": str(train_emb_path),
        "embeddings_val": str(val_emb_path),
    }
    if holdouts:
        summary["holdout_eval"] = _eval_holdouts(
            model,
            holdout_parquets=holdouts,
            artifact=adapter_payload,
            device=args.device,
            concat_original_features=bool(args.concat_original_features),
            selected_threshold=selected_threshold,
            feature_cols=feature_cols,
            emb_dir=emb_dir,
        )
    (lgbm_out / "retrain_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (lgbm_out / "feature_cols.json").write_text(
        json.dumps({"feature_cols": feature_cols, "n_features": len(feature_cols)}, indent=2),
        encoding="utf-8",
    )
    try:
        import joblib

        joblib.dump(model, lgbm_out / "lgbm_student.joblib")
    except Exception as e:
        print(f"[warn] joblib save skipped: {e}", file=sys.stderr)
    model.booster_.save_model(str(lgbm_out / "lgbm_student.txt"))

    print(json.dumps(summary, indent=2))
    print(f"Saved retrained student artifacts to {lgbm_out}", flush=True)


if __name__ == "__main__":
    main()
