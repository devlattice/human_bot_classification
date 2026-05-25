#!/usr/bin/env python3
"""
Evaluate DANN ``p_bot`` scores against ground-truth labels from a Parquet file.

Assumes ``--probs-npz`` rows align 1:1 with ``--labels-parquet`` row order (same as when
features were exported to ``holdout_1.npz``).

Writes metrics, per-row CSV, and plots under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plots. Install: pip install matplotlib"
    ) from e

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DANN p_bot vs Parquet labels + plots.")
    p.add_argument(
        "--probs-npz",
        type=Path,
        required=True,
        help="npz with key p_bot (from infer_dann.py)",
    )
    p.add_argument(
        "--labels-parquet",
        type=Path,
        required=True,
        help="Parquet with binary label column (same row order as features)",
    )
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--features-npz",
        type=Path,
        default=None,
        help="Optional: npz with X; must match row count with probs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory (e.g. workspace/DANN/infer/test)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probs_path = args.probs_npz.expanduser().resolve()
    lab_path = args.labels_parquet.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(probs_path, allow_pickle=False)
    if "p_bot" not in z.files:
        raise SystemExit(f"{probs_path}: expected key 'p_bot', got {z.files}")
    p_bot = np.asarray(z["p_bot"], dtype=np.float64).reshape(-1)

    df = pd.read_parquet(lab_path)
    if args.label_col not in df.columns:
        raise SystemExit(f"Missing column {args.label_col!r}; have {list(df.columns)}")
    y = pd.to_numeric(df[args.label_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(y).any():
        raise SystemExit(f"NaN in {args.label_col}")
    y = (y > 0.5).astype(np.int64)

    if args.features_npz is not None:
        fz = np.load(Path(args.features_npz).expanduser().resolve(), allow_pickle=False)
        if "X" in fz.files and fz["X"].shape[0] != len(p_bot):
            raise SystemExit(
                f"Row count mismatch: probs {len(p_bot)} vs features {fz['X'].shape[0]}"
            )

    if len(p_bot) != len(y):
        raise SystemExit(
            f"Row count mismatch: p_bot {len(p_bot)} vs labels {len(y)}. "
            "Ensure Parquet row order matches infer input."
        )

    y_pred = (p_bot >= args.threshold).astype(np.int64)

    acc = float(accuracy_score(y, y_pred))
    try:
        roc_auc = float(roc_auc_score(y, p_bot))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y, p_bot))
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["class_0", "class_1"],
        zero_division=0,
    )

    metrics = {
        "n": int(len(y)),
        "threshold": float(args.threshold),
        "accuracy": acc,
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "confusion_matrix": cm.tolist(),
        "labels_parquet": str(lab_path),
        "probs_npz": str(probs_path),
        "mean_p_bot_class0": float(p_bot[y == 0].mean()) if (y == 0).any() else None,
        "mean_p_bot_class1": float(p_bot[y == 1].mean()) if (y == 1).any() else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per = pd.DataFrame(
        {
            "y_true": y,
            "p_bot": p_bot,
            "y_pred": y_pred,
        }
    )
    per.to_csv(out_dir / "per_row.csv", index=False)

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    # --- plots ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    for (j, i), v in np.ndenumerate(cm):
        ax.text(i, j, str(int(v)), ha="center", va="center", color="black", fontsize=14)
    ax.set_title("Confusion matrix (threshold=%.2f)" % args.threshold)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p_bot)
        ax.plot(fpr, tpr, label="ROC (AUC=%.4f)" % roc_auc)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(np.unique(y)) > 1:
        prec, rec, _ = precision_recall_curve(y, p_bot)
        ax.plot(rec, prec, label="PR (AP=%.4f)" % pr_auc)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_bot[y == 0], bins=40, alpha=0.6, label="y=0", density=True, color="C0")
    ax.hist(p_bot[y == 1], bins=40, alpha=0.6, label="y=1", density=True, color="C1")
    ax.axvline(args.threshold, color="k", linestyle="--", label="threshold")
    ax.set_xlabel("p_bot")
    ax.set_ylabel("density")
    ax.set_title("p_bot distribution by true label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p_bot_hist_by_class.png", dpi=120)
    plt.close(fig)

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
