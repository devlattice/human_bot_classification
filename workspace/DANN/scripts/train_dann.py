#!/usr/bin/env python3
"""
Train DANN on tabular features: labeled source (human/bot) + unlabeled target.

Example:
  python train_dann.py --demo
  python train_dann.py --source-npz /path/source.npz --target-npz /path/target.npz --out /path/ckpt.pt

Expected npz keys:
  source: X (float32, [N, D]), y (int or float {0,1}, [N])
  source_val (optional): X (float32, [N_val, D]), y (int or float {0,1}, [N_val])
  target: X (float32, [M, D])
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DANN

from _threshold_metrics import (
    epoch_rank_tuple,
    metrics_at_threshold,
    select_threshold_bot_recall_under_human_fpr_cap,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logistic_lambda(progress: float, gamma: float = 10.0) -> float:
    """Maps progress in [0, 1] to [0, 1) via 2/(1+exp(-gamma*p)) - 1."""
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def load_npz_source(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.load(path, allow_pickle=False)
    if "X" not in z.files or "y" not in z.files:
        raise ValueError(f"{path}: expected keys 'X' and 'y', got {z.files}")
    X = np.asarray(z["X"], dtype=np.float32)
    y = np.asarray(z["y"]).reshape(-1)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.int64)
    y = y.astype(np.float32)
    seat: np.ndarray | None = None
    if "seat_bucket" in z.files:
        seat = np.asarray(z["seat_bucket"], dtype=np.int64).reshape(-1)
        if seat.shape[0] != X.shape[0]:
            raise ValueError(f"{path}: seat_bucket length {seat.shape[0]} != X rows {X.shape[0]}")
    return X, y, seat


def load_npz_target(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if "X" not in z.files:
        raise ValueError(f"{path}: expected key 'X', got {z.files}")
    return np.asarray(z["X"], dtype=np.float32)


def make_demo_data(
    n_source: int = 800,
    n_target: int = 600,
    dim: int = 16,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toy shift: target means shifted; labels from linear boundary on source."""
    rng = np.random.default_rng(seed)
    # Source: two classes with separated means
    n0 = n_source // 2
    n1 = n_source - n0
    mu0 = np.zeros(dim, dtype=np.float32)
    mu1 = np.ones(dim, dtype=np.float32) * 1.5
    X0 = rng.standard_normal((n0, dim)).astype(np.float32) * 0.5 + mu0
    X1 = rng.standard_normal((n1, dim)).astype(np.float32) * 0.5 + mu1
    Xs = np.vstack([X0, X1])
    ys = np.concatenate([np.zeros(n0, dtype=np.float32), np.ones(n1, dtype=np.float32)])
    idx = rng.permutation(n_source)
    Xs, ys = Xs[idx], ys[idx]

    # Target: shift + slightly different cov (domain shift)
    shift = rng.standard_normal(dim).astype(np.float32) * 0.4
    Xt = rng.standard_normal((n_target, dim)).astype(np.float32) * 0.6 + shift
    # Mixture of two components (unlabeled) — not used for training labels
    Xt[: n_target // 2] += mu0 * 0.8
    Xt[n_target // 2 :] += mu1 * 0.8
    idx_t = rng.permutation(n_target)
    Xt = Xt[idx_t]
    return Xs, ys, Xt


def fit_apply_scaler(
    X_train: np.ndarray, *rest: np.ndarray
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def tr(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    scaled = tuple(tr(x) for x in (X_train,) + rest)
    return scaled, mean, std


def train_one_epoch(
    model: DANN,
    opt: torch.optim.Optimizer,
    loader_src: DataLoader,
    loader_tgt: DataLoader,
    device: torch.device,
    lambda_grl: float,
    task_loss: nn.Module,
    dom_loss: nn.Module,
    *,
    nuis_loss: nn.Module | None,
    nuis_weight: float,
) -> Dict[str, float]:
    model.train()
    model.set_grl_lambda(lambda_grl)
    it_src = iter(loader_src)
    it_tgt = iter(loader_tgt)
    n_batches = min(len(loader_src), len(loader_tgt))
    total_Ly = 0.0
    total_Ld = 0.0
    total_Ln = 0.0
    n_dom = 0

    for _ in range(n_batches):
        try:
            b_src = next(it_src)
            xt = next(it_tgt)[0]
        except StopIteration:
            break
        xs = b_src[0].to(device)
        ys = b_src[1].to(device)
        xt = xt.to(device)

        opt.zero_grad(set_to_none=True)
        out_s = model(xs)
        out_t = model(xt)

        Ly = task_loss(out_s.task_logits, ys)
        # Domain: 0 = source, 1 = target
        dom_logits = torch.cat([out_s.domain_logits, out_t.domain_logits], dim=0)
        dom_labels = torch.cat(
            [
                torch.zeros(xs.size(0), device=device),
                torch.ones(xt.size(0), device=device),
            ],
            dim=0,
        )
        Ld = dom_loss(dom_logits, dom_labels)

        loss = Ly + Ld
        if (
            nuis_loss is not None
            and nuis_weight > 0.0
            and out_s.nuisance_logits is not None
            and len(b_src) > 2
        ):
            seat_b = b_src[2].to(device).long()
            Ln = nuis_loss(out_s.nuisance_logits, seat_b)
            loss = loss + nuis_weight * Ln
            total_Ln += float(Ln.detach().item())

        loss.backward()
        opt.step()

        total_Ly += float(Ly.detach().item())
        total_Ld += float(Ld.detach().item())
        n_dom += 1

    if n_dom == 0:
        return {"Ly": 0.0, "Ld": 0.0, "Ln": 0.0}
    out = {"Ly": total_Ly / n_dom, "Ld": total_Ld / n_dom}
    if nuis_loss is not None and nuis_weight > 0.0:
        out["Ln"] = total_Ln / n_dom
    else:
        out["Ln"] = 0.0
    return out


@torch.no_grad()
def eval_task_accuracy(
    model: DANN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        xs = batch[0].to(device)
        ys = batch[1].to(device)
        logits = model(xs).task_logits
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((pred == ys).sum().item())
        total += xs.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def collect_val_probs_labels(
    model: DANN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """All validation rows: y float {0,1}, p_bot in [0,1]."""
    model.eval()
    ys_list: list[torch.Tensor] = []
    ps_list: list[torch.Tensor] = []
    for batch in loader:
        xs = batch[0].to(device)
        ys = batch[1]
        logits = model(xs).task_logits
        p = torch.sigmoid(logits).reshape(-1)
        ys_list.append(ys.reshape(-1).detach().cpu())
        ps_list.append(p.detach().cpu())
    if not ys_list:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    y_np = torch.cat(ys_list, dim=0).numpy().astype(np.float32)
    p_np = torch.cat(ps_list, dim=0).numpy().astype(np.float32)
    return y_np, p_np


@torch.no_grad()
def eval_domain_confusion(
    model: DANN,
    loader_src: DataLoader,
    loader_tgt: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Domain confusion on source-vs-target features.

    Returns (domain_bce_loss, domain_accuracy). Higher loss / lower accuracy means
    more confusion between source and target domains.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct = 0
    it_src = iter(loader_src)
    it_tgt = iter(loader_tgt)
    n_batches = min(len(loader_src), len(loader_tgt))
    for _ in range(n_batches):
        try:
            xs = next(it_src)[0].to(device)
            xt = next(it_tgt)[0].to(device)
        except StopIteration:
            break
        out_s = model(xs)
        out_t = model(xt)
        dom_logits = torch.cat([out_s.domain_logits, out_t.domain_logits], dim=0)
        dom_labels = torch.cat(
            [
                torch.zeros(xs.size(0), device=device),
                torch.ones(xt.size(0), device=device),
            ],
            dim=0,
        )
        total_loss += float(loss_fn(dom_logits, dom_labels).item())
        pred = (torch.sigmoid(dom_logits) >= 0.5).float()
        correct += int((pred == dom_labels).sum().item())
        total += int(dom_labels.numel())
    if total == 0:
        return 0.0, 0.0
    return total_loss / float(total), correct / float(total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DANN (tabular features).")
    p.add_argument("--demo", action="store_true", help="Run on synthetic data.")
    p.add_argument("--source-npz", type=Path, default=None, help="npz with X, y")
    p.add_argument(
        "--source-val-npz",
        type=Path,
        default=None,
        help="optional held-out validation npz with X, y; skips --val-frac split",
    )
    p.add_argument(
        "--extra-val-npz",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional extra labeled validation npz (X,y). Repeatable. "
            "Used by multi_objective_generalization selection metric."
        ),
    )
    p.add_argument("--target-npz", type=Path, default=None, help="npz with X")
    p.add_argument("--out", type=Path, default=None, help="Checkpoint .pt path")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=1.0, help="Max GRL strength.")
    p.add_argument("--lambda-gamma", type=float, default=10.0, help="Logistic ramp sharpness.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu). Use --device cpu to force CPU.",
    )
    p.add_argument(
        "--val-selection-metric",
        type=str,
        default="bot_recall_at_human_fpr_cap",
        choices=(
            "val_acc",
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ),
        help="Checkpoint selection: val_acc @0.5, FPR-capped bot recall, or same with domain-confusion tie-break.",
    )
    p.add_argument(
        "--target-human-fpr",
        type=float,
        default=0.05,
        help="Human FPR cap when --val-selection-metric=bot_recall_at_human_fpr_cap.",
    )
    p.add_argument(
        "--threshold-grid-size",
        type=int,
        default=401,
        help="Number of thresholds in [0,1] for val calibration sweep.",
    )
    p.add_argument(
        "--threshold-tie-ref",
        type=float,
        default=0.5,
        help="When multiple thresholds tie, prefer closest to this value.",
    )
    p.add_argument(
        "--hybrid-nuisance-seat",
        action="store_true",
        help="Hybrid model path: nuisance seat-bucket head on GRL features (needs seat_bucket in labeled npz).",
    )
    p.add_argument(
        "--hybrid-nuisance-weight",
        type=float,
        default=0.5,
        help="Scale for nuisance CE loss (multiplied by the same GRL ramp as domain loss).",
    )
    p.add_argument(
        "--n-seat-buckets",
        type=int,
        default=9,
        help="Classes for seat_bucket (default 9 for player counts 2..10).",
    )
    return p.parse_args()


def _resolve_device(explicit: str | None) -> torch.device:
    if explicit is not None and explicit.strip() != "":
        if explicit == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    if args.demo:
        Xs, ys, Xt = make_demo_data(seed=args.seed)
        seat_all: np.ndarray | None = None
    else:
        if args.source_npz is None or args.target_npz is None:
            raise SystemExit("--source-npz and --target-npz required unless --demo")
        Xs, ys, seat_all = load_npz_source(args.source_npz)
        Xt = load_npz_target(args.target_npz)

    if Xs.shape[1] != Xt.shape[1]:
        raise SystemExit(f"Feature dim mismatch: source {Xs.shape[1]} vs target {Xt.shape[1]}")

    # Use explicit held-out source val when provided; otherwise split source train.
    seat_tr: np.ndarray | None
    seat_val: np.ndarray | None
    if args.source_val_npz is not None:
        X_tr, y_tr = Xs, ys
        seat_tr = seat_all
        X_val, y_val, seat_val = load_npz_source(args.source_val_npz)
        if X_tr.shape[1] != X_val.shape[1]:
            raise SystemExit(
                f"Feature dim mismatch: source train {X_tr.shape[1]} vs source val {X_val.shape[1]}"
            )
    else:
        n = Xs.shape[0]
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        n_val = int(round(n * args.val_frac))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        X_tr, y_tr = Xs[train_idx], ys[train_idx]
        X_val, y_val = Xs[val_idx], ys[val_idx]
        seat_tr = seat_all[train_idx] if seat_all is not None else None
        seat_val = seat_all[val_idx] if seat_all is not None else None

    if bool(args.hybrid_nuisance_seat) and seat_tr is None:
        raise SystemExit(
            "--hybrid-nuisance-seat requires `seat_bucket` in labeled source npz "
            "(export with export_parquet_to_source_npz.py --seat-bucket-col n_players_max)."
        )

    (X_tr, X_val, Xt), mean, std = fit_apply_scaler(X_tr, X_val, Xt)

    extra_vals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, p in enumerate(args.extra_val_npz):
        xp, yp, _seat_e = load_npz_source(p)
        if X_tr.shape[1] != xp.shape[1]:
            raise SystemExit(
                f"Feature dim mismatch: source train {X_tr.shape[1]} vs extra val[{i}] {xp.shape[1]} ({p})"
            )
        xp = ((xp - mean) / std).astype(np.float32)
        extra_vals.append((str(Path(p).expanduser().resolve()), xp, yp))

    if bool(args.hybrid_nuisance_seat) and seat_tr is not None:
        ds_tr = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr),
            torch.from_numpy(seat_tr),
        )
    else:
        ds_tr = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr),
        )
    if seat_val is not None:
        ds_val = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(seat_val),
        )
    else:
        ds_val = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
    ds_tgt = TensorDataset(torch.from_numpy(Xt))

    if X_tr.shape[0] == 0:
        raise SystemExit("Source train has zero rows.")
    if X_val.shape[0] == 0:
        raise SystemExit("Validation source has zero rows.")
    if Xt.shape[0] == 0:
        raise SystemExit("Target npz has zero rows.")

    loader_src = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    loader_tgt = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_tgt_eval = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=False, drop_last=False)
    extra_val_loaders: list[tuple[str, DataLoader]] = []
    for name, xev, yev in extra_vals:
        ds_ev = TensorDataset(torch.from_numpy(xev), torch.from_numpy(yev))
        extra_val_loaders.append((name, DataLoader(ds_ev, batch_size=args.batch_size, shuffle=False)))

    if len(loader_src) == 0 or len(loader_tgt) == 0:
        raise SystemExit(
            f"No training batches (drop_last=True). source_train={X_tr.shape[0]} "
            f"target={Xt.shape[0]} batch_size={args.batch_size} — lower --batch-size or add rows."
        )

    n_batch = min(len(loader_src), len(loader_tgt))
    print(
        f"[train_dann] device={device}  D={Xs.shape[1]}  "
        f"source train/val={X_tr.shape[0]}/{X_val.shape[0]}  target={Xt.shape[0]}  "
        f"extra_val_domains={len(extra_val_loaders)}  batches_per_epoch={n_batch}  epochs={args.epochs}",
        flush=True,
    )

    in_dim = Xs.shape[1]
    use_nuis = bool(args.hybrid_nuisance_seat) and seat_tr is not None
    model = DANN(
        in_dim=in_dim,
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        grl_lambda=0.0,
        use_nuisance_seat=use_nuis,
        n_seat_buckets=int(args.n_seat_buckets),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    task_loss = nn.BCEWithLogitsLoss()
    dom_loss = nn.BCEWithLogitsLoss()
    nuis_loss: nn.CrossEntropyLoss | None = nn.CrossEntropyLoss() if use_nuis else None

    best_rank: Tuple[float, ...] | None = None
    best_val_acc_seen = -1.0
    best_state: Dict[str, Any] | None = None
    log_every = max(1, args.epochs // 10)
    sel_metric = str(args.val_selection_metric)

    for epoch in range(args.epochs):
        progress = (epoch + 1) / max(args.epochs, 1)
        lam = args.lambda_max * logistic_lambda(progress, gamma=args.lambda_gamma)
        nuis_w = float(args.hybrid_nuisance_weight) * lam if use_nuis else 0.0
        metrics = train_one_epoch(
            model,
            opt,
            loader_src,
            loader_tgt,
            device,
            lambda_grl=lam,
            task_loss=task_loss,
            dom_loss=dom_loss,
            nuis_loss=nuis_loss,
            nuis_weight=nuis_w,
        )
        val_acc = eval_task_accuracy(model, loader_val, device)
        best_val_acc_seen = max(best_val_acc_seen, val_acc)
        y_val_np, p_val_np = collect_val_probs_labels(model, loader_val, device)
        sweep = None
        dom_conf_loss = None
        dom_conf_acc = None
        if sel_metric in (
            "bot_recall_at_human_fpr_cap",
            "bot_recall_at_human_fpr_cap_then_domain_confusion",
            "multi_objective_generalization",
        ):
            sweep = select_threshold_bot_recall_under_human_fpr_cap(
                y_val_np,
                p_val_np,
                human_fpr_cap=float(args.target_human_fpr),
                grid_size=int(args.threshold_grid_size),
                tie_ref=float(args.threshold_tie_ref),
            )
        if sel_metric == "bot_recall_at_human_fpr_cap_then_domain_confusion":
            dom_conf_loss, dom_conf_acc = eval_domain_confusion(
                model, loader_val, loader_tgt_eval, device
            )
        multi_obj = None
        if sel_metric == "multi_objective_generalization":
            dom_conf_loss, dom_conf_acc = eval_domain_confusion(
                model, loader_val, loader_tgt_eval, device
            )
            sweeps_all: list[dict[str, Any]] = []
            if sweep is not None:
                sweeps_all.append(sweep)
            for _, ld in extra_val_loaders:
                y_e, p_e = collect_val_probs_labels(model, ld, device)
                s_e = select_threshold_bot_recall_under_human_fpr_cap(
                    y_e,
                    p_e,
                    human_fpr_cap=float(args.target_human_fpr),
                    grid_size=int(args.threshold_grid_size),
                    tie_ref=float(args.threshold_tie_ref),
                )
                sweeps_all.append(s_e)
            all_feasible = bool(sweeps_all) and all(bool(s.get("feasible", False)) for s in sweeps_all)
            bot_recalls = [float(s.get("bot_recall", 0.0)) for s in sweeps_all]
            human_fprs = [float(s.get("human_fpr", 1.0)) for s in sweeps_all]
            multi_obj = {
                "all_feasible": all_feasible,
                "worst_bot_recall": min(bot_recalls) if bot_recalls else 0.0,
                "mean_bot_recall": float(np.mean(bot_recalls)) if bot_recalls else 0.0,
                "mean_human_fpr": float(np.mean(human_fprs)) if human_fprs else 1.0,
            }
        rank = epoch_rank_tuple(sel_metric, val_acc, sweep, dom_conf_loss, multi_objective=multi_obj)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % log_every == 0 or epoch == 0:
            extra = ""
            if sweep is not None:
                extra = (
                    f"  thr={sweep['threshold']:.4f}  human_fpr={sweep['human_fpr']:.4f}  "
                    f"bot_recall={sweep['bot_recall']:.4f}  feasible={sweep['feasible']}"
                )
            if dom_conf_loss is not None and dom_conf_acc is not None:
                extra += (
                    f"  domain_conf_loss={dom_conf_loss:.4f}  "
                    f"domain_acc={dom_conf_acc:.4f}"
                )
            if multi_obj is not None:
                extra += (
                    f"  worst_bot_recall={multi_obj['worst_bot_recall']:.4f}"
                    f"  mean_bot_recall={multi_obj['mean_bot_recall']:.4f}"
                    f"  mean_human_fpr={multi_obj['mean_human_fpr']:.4f}"
                    f"  all_feasible={multi_obj['all_feasible']}"
                )
            ln_s = f"  Ln={metrics.get('Ln', 0.0):.4f}" if use_nuis else ""
            print(
                f"[train_dann] epoch {epoch+1}/{args.epochs}  lambda={lam:.4f}  "
                f"Ly={metrics['Ly']:.4f}  Ld={metrics['Ld']:.4f}{ln_s}  val_acc@0.5={val_acc:.4f}"
                f"{extra}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = args.out
    if out_path is None:
        out_path = _SCRIPTS.parent / "artifacts" / "dann_checkpoint.pt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_final, p_final = collect_val_probs_labels(model, loader_val, device)
    final_sweep = None
    if sel_metric in (
        "bot_recall_at_human_fpr_cap",
        "bot_recall_at_human_fpr_cap_then_domain_confusion",
        "multi_objective_generalization",
    ):
        final_sweep = select_threshold_bot_recall_under_human_fpr_cap(
            y_final,
            p_final,
            human_fpr_cap=float(args.target_human_fpr),
            grid_size=int(args.threshold_grid_size),
            tie_ref=float(args.threshold_tie_ref),
        )

    if sel_metric == "val_acc":
        best_score = float(best_val_acc_seen)
        selected_threshold = 0.5
        val_human_fpr_at_t, val_bot_recall_at_t, val_acc_at_selected_t = metrics_at_threshold(
            y_final, p_final, 0.5
        )
        val_threshold_feasible = True
    else:
        assert final_sweep is not None
        selected_threshold = float(final_sweep["threshold"])
        val_human_fpr_at_t = float(final_sweep["human_fpr"])
        val_bot_recall_at_t = float(final_sweep["bot_recall"])
        _, _, val_acc_at_selected_t = metrics_at_threshold(y_final, p_final, selected_threshold)
        val_threshold_feasible = bool(final_sweep.get("feasible"))
        best_score = float(val_bot_recall_at_t)
    extra_val_metrics: list[dict[str, Any]] = []
    if sel_metric == "multi_objective_generalization":
        assert final_sweep is not None
        sweeps = [final_sweep]
        extra_val_metrics.append(
            {
                "name": "source_val",
                "human_fpr": float(final_sweep["human_fpr"]),
                "bot_recall": float(final_sweep["bot_recall"]),
                "threshold": float(final_sweep["threshold"]),
                "feasible": bool(final_sweep.get("feasible", False)),
            }
        )
        for name, ld in extra_val_loaders:
            y_e, p_e = collect_val_probs_labels(model, ld, device)
            s_e = select_threshold_bot_recall_under_human_fpr_cap(
                y_e,
                p_e,
                human_fpr_cap=float(args.target_human_fpr),
                grid_size=int(args.threshold_grid_size),
                tie_ref=float(args.threshold_tie_ref),
            )
            sweeps.append(s_e)
            extra_val_metrics.append(
                {
                    "name": name,
                    "human_fpr": float(s_e["human_fpr"]),
                    "bot_recall": float(s_e["bot_recall"]),
                    "threshold": float(s_e["threshold"]),
                    "feasible": bool(s_e.get("feasible", False)),
                }
            )
        best_score = float(min(float(s["bot_recall"]) for s in sweeps))
    final_domain_conf_loss, final_domain_conf_acc = eval_domain_confusion(
        model, loader_val, loader_tgt_eval, device
    )

    meta: Dict[str, Any] = {
        "in_dim": in_dim,
        "feat_dim": args.feat_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_val_acc": float(best_val_acc_seen),
        "val_selection_metric": sel_metric,
        "threshold_policy": (
            "val_recalibrated_each_run_bot_recall_under_human_fpr_cap"
            if sel_metric
            in (
                "bot_recall_at_human_fpr_cap",
                "bot_recall_at_human_fpr_cap_then_domain_confusion",
                "multi_objective_generalization",
            )
            else "fixed_threshold_0_5_for_val_acc_mode"
        ),
        "target_human_fpr": float(args.target_human_fpr),
        "threshold_grid_size": int(args.threshold_grid_size),
        "threshold_tie_ref": float(args.threshold_tie_ref),
        "selected_threshold": float(selected_threshold),
        "val_human_fpr_at_selected_threshold": float(val_human_fpr_at_t),
        "val_bot_recall_at_selected_threshold": float(val_bot_recall_at_t),
        "val_acc_at_selected_threshold": float(val_acc_at_selected_t),
        "val_threshold_feasible": bool(val_threshold_feasible),
        "best_score": float(best_score),
        "domain_confusion_loss_source_val_vs_target": float(final_domain_conf_loss),
        "domain_confusion_acc_source_val_vs_target": float(final_domain_conf_acc),
        "n_extra_val_domains": int(len(extra_val_loaders)),
        "extra_val_metrics": extra_val_metrics,
        "epochs": args.epochs,
        "seed": args.seed,
        "use_nuisance_seat": bool(use_nuis),
        "n_seat_buckets": int(args.n_seat_buckets),
        "hybrid_nuisance_weight": float(args.hybrid_nuisance_weight),
    }
    if final_sweep is not None:
        meta["threshold_selection_rule"] = str(final_sweep.get("selection_rule", ""))
    elif sel_metric == "val_acc":
        meta["threshold_selection_rule"] = "val_acc_at_threshold_0.5"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": meta,
            "scaler_mean": mean,
            "scaler_std": std,
        },
        out_path,
    )
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
