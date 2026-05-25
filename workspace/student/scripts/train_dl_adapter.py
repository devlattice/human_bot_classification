#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace.model.scripts.lgbm import _threshold_sweep, evaluate


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:  # pragma: no cover
        raise SystemExit("PyTorch is required. Install torch first.") from e
    return torch, nn, F, DataLoader, TensorDataset


def _read_feature_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    cols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols.append(s)
    return cols


def _load_labeled(path: Path, feature_cols: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(path.expanduser().resolve())
    if "label" not in df.columns:
        raise ValueError(f"{path}: missing label column")
    if feature_cols is None:
        cols = [c for c in df.columns if c != "label"]
    else:
        miss = [c for c in feature_cols if c not in df.columns]
        if miss:
            raise ValueError(f"{path}: missing {len(miss)} features, e.g. {miss[:3]}")
        cols = list(feature_cols)
    return df, cols


def _load_unlabeled(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path.expanduser().resolve())
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{path}: missing {len(miss)} features, e.g. {miss[:3]}")
    return df


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return cls(mean=mu.astype(np.float32), std=sd.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z.astype(np.float32)


def _sigmoid_ramp(step: int, total: int, gamma: float, max_w: float) -> float:
    if total <= 1:
        p = 1.0
    else:
        p = float(step) / float(total - 1)
    return float(max_w * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


def _sweep_selected_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_human_fpr: float,
    grid_size: int,
    tie_ref: float,
) -> dict[str, float | bool]:
    sweep = _threshold_sweep(
        y_true=y_true,
        y_score=y_score,
        target_human_fpr=target_human_fpr,
        grid_size=grid_size,
        threshold_tie_ref=tie_ref,
    )
    sel = sweep["selected_metrics"]
    human_fpr = float(sel["human_fpr"])
    bot_recall = float(sel["bot_recall"])
    return {
        "threshold": float(sweep["selected_threshold"]),
        "human_fpr": human_fpr,
        "bot_recall": bot_recall,
        "feasible": bool(human_fpr <= float(target_human_fpr) + 1e-12),
    }


def _epoch_rank_tuple(
    metric: str,
    *,
    main_sel: dict[str, float | bool],
    domain_conf_loss: float,
    domain_conf_score: float,
    multi_obj: dict[str, float | bool] | None,
    val_score_legacy: float,
    domain_selection_weight: float,
) -> tuple[float, ...]:
    if metric == "val_score_plus_domain_confusion":
        return (float(val_score_legacy + float(domain_selection_weight) * domain_conf_score),)
    feas = 1.0 if bool(main_sel.get("feasible", False)) else 0.0
    br = float(main_sel.get("bot_recall", 0.0))
    hf = float(main_sel.get("human_fpr", 1.0))
    if metric == "bot_recall_at_human_fpr_cap":
        return (feas, br, -hf)
    if metric == "bot_recall_at_human_fpr_cap_then_domain_confusion":
        return (feas, br, -hf, float(domain_conf_loss))
    if metric == "multi_objective_generalization":
        if multi_obj is None:
            return (float("-inf"),)
        return (
            1.0 if bool(multi_obj.get("all_feasible", False)) else 0.0,
            float(multi_obj.get("worst_bot_recall", 0.0)),
            float(multi_obj.get("mean_bot_recall", 0.0)),
            -float(multi_obj.get("mean_human_fpr", 1.0)),
            float(domain_conf_loss),
        )
    raise ValueError(f"unknown --val-selection-metric: {metric}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train domain-adaptive DL adapter (DANN-style) on tabular features.")
    p.add_argument("--source-train", type=Path, required=True)
    p.add_argument("--source-val", type=Path, required=True)
    p.add_argument("--target-unlabeled", type=Path, required=True)
    p.add_argument("--feature-cols-file", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=8)
    # Defaults match adapter_grid_domain_conf_v2 grid_summary best_trial (trial 7).
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--lambda-domain-max", type=float, default=0.05)
    p.add_argument("--lambda-domain-gamma", type=float, default=8.0)
    p.add_argument("--target-human-fpr", type=float, default=0.05)
    p.add_argument("--threshold-grid-size", type=int, default=1001)
    p.add_argument("--threshold-tie-ref", type=float, default=0.5)
    p.add_argument(
        "--extra-val-labeled",
        action="append",
        type=Path,
        default=[],
        help="Optional extra labeled validation parquet(s), repeatable; used in multi-objective selection.",
    )
    p.add_argument(
        "--val-selection-metric",
        type=str,
        default="val_score_plus_domain_confusion",
        choices=(
            "val_score_plus_domain_confusion",
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ),
    )
    p.add_argument(
        "--domain-selection-weight",
        type=float,
        default=0.6,
        help="Weight for domain-confusion score in model selection: selection = val_score + w*domain_confusion_score",
    )
    p.add_argument(
        "--domain-eval-target-rows",
        type=int,
        default=10000,
        help="Rows from target used for epoch domain eval (0=match val rows).",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    torch, nn, F, DataLoader, TensorDataset = _require_torch()
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    fcols = _read_feature_list(args.feature_cols_file)
    src_train_df, feature_cols = _load_labeled(args.source_train, fcols)
    src_val_df, _ = _load_labeled(args.source_val, feature_cols)
    tgt_df = _load_unlabeled(args.target_unlabeled, feature_cols)
    extra_val_dfs: list[tuple[str, pd.DataFrame]] = []
    for p_extra in args.extra_val_labeled:
        ex_df, _ = _load_labeled(Path(p_extra), feature_cols)
        extra_val_dfs.append((str(Path(p_extra).expanduser().resolve()), ex_df))

    x_train_raw = src_train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = src_train_df["label"].to_numpy(dtype=np.int64)
    x_val_raw = src_val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = src_val_df["label"].to_numpy(dtype=np.int64)
    x_tgt_raw = tgt_df[feature_cols].to_numpy(dtype=np.float32)

    stdz = Standardizer.fit(x_train_raw)
    x_train = stdz.transform(x_train_raw)
    x_val = stdz.transform(x_val_raw)
    x_tgt = stdz.transform(x_tgt_raw)
    x_extra_vals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, ex_df in extra_val_dfs:
        x_ex = stdz.transform(ex_df[feature_cols].to_numpy(dtype=np.float32))
        y_ex = ex_df["label"].to_numpy(dtype=np.int64)
        x_extra_vals.append((name, x_ex, y_ex))
    n_dom_eval = int(args.domain_eval_target_rows) if int(args.domain_eval_target_rows) > 0 else int(len(x_val))
    n_dom_eval = max(1, min(n_dom_eval, int(len(x_tgt))))
    tgt_eval_idx = rng.choice(len(x_tgt), size=n_dom_eval, replace=False)
    x_tgt_eval = x_tgt[tgt_eval_idx]

    src_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    tgt_ds = TensorDataset(torch.from_numpy(x_tgt))
    src_loader = DataLoader(src_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, coeff):
            ctx.coeff = coeff
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.coeff * grad_output, None

    class AdapterNet(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.task_head = nn.Linear(embed_dim, 1)
            self.domain_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def embed(self, x):
            return self.encoder(x)

        def task_logit(self, z):
            return self.task_head(z).squeeze(-1)

        def domain_logit(self, z, coeff: float):
            zr = GradReverse.apply(z, coeff)
            return self.domain_head(zr).squeeze(-1)

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)

    model = AdapterNet(
        in_dim=len(feature_cols),
        hidden_dim=int(args.hidden_dim),
        embed_dim=int(args.embed_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce = nn.BCEWithLogitsLoss()

    best_state: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    history: list[dict[str, Any]] = []
    total_steps = max(1, int(args.epochs) * max(1, min(len(src_loader), len(tgt_loader))))
    step = 0

    for epoch in range(int(args.epochs)):
        model.train()
        src_it = iter(src_loader)
        tgt_it = iter(tgt_loader)
        n_iter = max(1, min(len(src_loader), len(tgt_loader)))
        losses = []
        for _ in range(n_iter):
            xs, ys = next(src_it)
            (xt,) = next(tgt_it)
            xs = xs.to(device)
            ys = ys.to(device=device, dtype=torch.float32)
            xt = xt.to(device)
            if epoch < int(args.warmup_epochs):
                lam = 0.0
            else:
                lam = _sigmoid_ramp(step, total_steps, float(args.lambda_domain_gamma), float(args.lambda_domain_max))

            zs = model.embed(xs)
            zt = model.embed(xt)
            task_loss = bce(model.task_logit(zs), ys)
            dom_s = model.domain_logit(zs, lam)
            dom_t = model.domain_logit(zt, lam)
            dom_y_s = torch.zeros_like(dom_s)
            dom_y_t = torch.ones_like(dom_t)
            dom_loss = 0.5 * (bce(dom_s, dom_y_s) + bce(dom_t, dom_y_t))
            loss = task_loss + dom_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val).to(device)
            zv = model.embed(xv)
            y_score = torch.sigmoid(model.task_logit(zv)).detach().cpu().numpy()
            # Domain eval without GRL: source(val) vs random target subset.
            xt_eval = torch.from_numpy(x_tgt_eval).to(device)
            zt_eval = model.embed(xt_eval)
            dom_s_eval = model.domain_head(zv).squeeze(-1)
            dom_t_eval = model.domain_head(zt_eval).squeeze(-1)
            dom_y_s_eval = torch.zeros_like(dom_s_eval)
            dom_y_t_eval = torch.ones_like(dom_t_eval)
            dom_loss_eval = 0.5 * (
                bce(dom_s_eval, dom_y_s_eval) + bce(dom_t_eval, dom_y_t_eval)
            )
            dom_prob = torch.sigmoid(torch.cat([dom_s_eval, dom_t_eval], dim=0)).detach().cpu().numpy()
            dom_true = np.concatenate(
                [
                    np.zeros(dom_s_eval.shape[0], dtype=np.int64),
                    np.ones(dom_t_eval.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            dom_pred = (dom_prob >= 0.5).astype(np.int64)
            dom_acc = float(np.mean(dom_pred == dom_true))
            # 1.0 is best (chance-level domain acc=0.5), 0.0 is worst.
            dom_confusion_score = float(max(0.0, 1.0 - 2.0 * abs(dom_acc - 0.5)))
        main_sel = _sweep_selected_metrics(
            y_true=y_val,
            y_score=y_score,
            target_human_fpr=float(args.target_human_fpr),
            grid_size=int(args.threshold_grid_size),
            tie_ref=float(args.threshold_tie_ref),
        )
        val_score = float(main_sel["bot_recall"]) - 0.25 * max(
            0.0, float(main_sel["human_fpr"]) - float(args.target_human_fpr)
        )
        selection_score = float(val_score + float(args.domain_selection_weight) * dom_confusion_score)
        per_domain_rows: list[dict[str, Any]] = [
            {
                "name": "source_val",
                "human_fpr": float(main_sel["human_fpr"]),
                "bot_recall": float(main_sel["bot_recall"]),
                "threshold": float(main_sel["threshold"]),
                "feasible": bool(main_sel["feasible"]),
            }
        ]
        for name, x_ex, y_ex in x_extra_vals:
            with torch.no_grad():
                xx = torch.from_numpy(x_ex).to(device)
                yy_score = torch.sigmoid(model.task_logit(model.embed(xx))).detach().cpu().numpy()
            sel_ex = _sweep_selected_metrics(
                y_true=y_ex,
                y_score=yy_score,
                target_human_fpr=float(args.target_human_fpr),
                grid_size=int(args.threshold_grid_size),
                tie_ref=float(args.threshold_tie_ref),
            )
            per_domain_rows.append(
                {
                    "name": name,
                    "human_fpr": float(sel_ex["human_fpr"]),
                    "bot_recall": float(sel_ex["bot_recall"]),
                    "threshold": float(sel_ex["threshold"]),
                    "feasible": bool(sel_ex["feasible"]),
                }
            )
        multi_obj = None
        if str(args.val_selection_metric) == "multi_objective_generalization":
            bot_recalls = [float(r["bot_recall"]) for r in per_domain_rows]
            human_fprs = [float(r["human_fpr"]) for r in per_domain_rows]
            multi_obj = {
                "all_feasible": bool(per_domain_rows) and all(bool(r["feasible"]) for r in per_domain_rows),
                "worst_bot_recall": min(bot_recalls) if bot_recalls else 0.0,
                "mean_bot_recall": float(np.mean(bot_recalls)) if bot_recalls else 0.0,
                "mean_human_fpr": float(np.mean(human_fprs)) if human_fprs else 1.0,
            }
        rank = _epoch_rank_tuple(
            str(args.val_selection_metric),
            main_sel=main_sel,
            domain_conf_loss=float(dom_loss_eval.detach().cpu().item()),
            domain_conf_score=dom_confusion_score,
            multi_obj=multi_obj,
            val_score_legacy=val_score,
            domain_selection_weight=float(args.domain_selection_weight),
        )
        if str(args.val_selection_metric) == "multi_objective_generalization" and multi_obj is not None:
            best_score_value = float(multi_obj["worst_bot_recall"])
        elif str(args.val_selection_metric) == "val_score_plus_domain_confusion":
            best_score_value = float(selection_score)
        else:
            best_score_value = float(main_sel["bot_recall"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_score": val_score,
            "domain_loss_eval": float(dom_loss_eval.detach().cpu().item()),
            "domain_acc_eval": dom_acc,
            "domain_confusion_score": dom_confusion_score,
            "selection_score": selection_score,
            "val_selection_metric": str(args.val_selection_metric),
            "best_score": best_score_value,
            "val_human_fpr": float(main_sel["human_fpr"]),
            "val_bot_recall": float(main_sel["bot_recall"]),
            "selected_threshold": float(main_sel["threshold"]),
            "val_threshold_feasible": bool(main_sel["feasible"]),
        }
        if multi_obj is not None:
            row.update(
                {
                    "all_feasible": bool(multi_obj["all_feasible"]),
                    "worst_bot_recall": float(multi_obj["worst_bot_recall"]),
                    "mean_bot_recall": float(multi_obj["mean_bot_recall"]),
                    "mean_human_fpr": float(multi_obj["mean_human_fpr"]),
                }
            )
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "main_sel": main_sel,
                "rank": list(rank),
                "best_score": best_score_value,
                "selection_score": selection_score,
                "domain_acc_eval": dom_acc,
                "domain_loss_eval": float(dom_loss_eval.detach().cpu().item()),
                "domain_confusion_score": dom_confusion_score,
                "val_score": val_score,
                "extra_val_metrics": per_domain_rows,
                "multi_objective": multi_obj,
            }

    if best_state is None:
        raise SystemExit("training failed: no best state")

    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val).to(device)
        yv_score = torch.sigmoid(model.task_logit(model.embed(xv))).detach().cpu().numpy()
    selected_threshold = float(best_state["main_sel"]["threshold"])
    yv_pred = (yv_score >= selected_threshold).astype(int)

    # Minimal evaluate-like report for adapter head itself.
    class _Tmp:
        def predict_proba(self, X):
            xx = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            with torch.no_grad():
                zz = model.embed(xx)
                pp = torch.sigmoid(model.task_logit(zz)).detach().cpu().numpy()
            return np.column_stack([1.0 - pp, pp])

    eval_val = evaluate(_Tmp(), x_val, y_val, threshold=selected_threshold)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "dl_adapter.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "mean": stdz.mean,
            "std": stdz.std,
            "embed_dim": int(args.embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "best_epoch": int(best_state["epoch"]),
            "selected_threshold": float(selected_threshold),
            "device": dev,
        },
        ckpt_path,
    )
    report = {
        "source_train": str(args.source_train.expanduser().resolve()),
        "source_val": str(args.source_val.expanduser().resolve()),
        "target_unlabeled": str(args.target_unlabeled.expanduser().resolve()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "train_rows": int(len(src_train_df)),
        "val_rows": int(len(src_val_df)),
        "target_rows": int(len(tgt_df)),
        "params": {
            "epochs": int(args.epochs),
            "warmup_epochs": int(args.warmup_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "embed_dim": int(args.embed_dim),
            "dropout": float(args.dropout),
            "lambda_domain_max": float(args.lambda_domain_max),
            "lambda_domain_gamma": float(args.lambda_domain_gamma),
            "domain_selection_weight": float(args.domain_selection_weight),
            "domain_eval_target_rows": int(n_dom_eval),
            "val_selection_metric": str(args.val_selection_metric),
            "target_human_fpr": float(args.target_human_fpr),
            "threshold_grid_size": int(args.threshold_grid_size),
            "threshold_tie_ref": float(args.threshold_tie_ref),
        },
        "best_epoch": int(best_state["epoch"]),
        "best_selection_score": float(best_state.get("selection_score", float("nan"))),
        "best_rank": best_state.get("rank"),
        "best_score": float(best_state.get("best_score", float("nan"))),
        "best_val_score": float(best_state.get("val_score", float("nan"))),
        "best_domain_acc_eval": float(best_state.get("domain_acc_eval", float("nan"))),
        "best_domain_loss_eval": float(best_state.get("domain_loss_eval", float("nan"))),
        "best_domain_confusion_score": float(
            best_state.get("domain_confusion_score", float("nan"))
        ),
        "n_extra_val_domains": int(len(x_extra_vals)),
        "extra_val_labeled": [name for name, _df in extra_val_dfs],
        "extra_val_metrics": best_state.get("extra_val_metrics", []),
        "multi_objective": best_state.get("multi_objective"),
        "selected_threshold": selected_threshold,
        "val_metrics_at_selected_threshold": eval_val,
        "history": history,
        "artifact": str(ckpt_path),
    }
    (out_dir / "adapter_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(ckpt_path), "best_epoch": int(best_state["epoch"]), "val": eval_val}, indent=2))


if __name__ == "__main__":
    main()

