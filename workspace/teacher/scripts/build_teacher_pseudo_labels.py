#!/usr/bin/env python3
"""
Build a pseudo-labeled parquet from unlabeled validator rows.

**Dual-teacher (default):**
  - DANN: raw tabular features (same column order as DANN source export).
  - SSL+LGBM: embedding parquet from ssl_embed (emb_* + originals), scored with lgbm_b_classifier.joblib.

**DANN-only (``--dann-only``):** SSL path is skipped; ``p_teacher = p_dann``. ``teacher_agreement`` uses a
**confidence margin** ``2 * |p_dann - 0.5|`` when a single checkpoint is used.

**Two-checkpoint DANN ensemble (``--dann-ckpt-b``):** run ``infer_dann`` twice on the same NPZ;
``p_dann = mean(p_a, p_b)``; optional columns ``p_dann_a``, ``p_dann_b``. Agreement combines seed
disagreement ``1 - |p_a - p_b|`` with either margin (DANN-only) or DANN–LGBM agreement (dual mode).

Row alignment (SSL mode only): ``--unlabeled-validator`` must match SSL
``embeddings/<basename>.parquet`` row order.

See ``workspace/teacher/docs/readme.md`` for weighting / confidence bands.

Run from repo root::

  PYTHONPATH=. python workspace/teacher/scripts/build_teacher_pseudo_labels.py --help
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DANN_SCRIPTS = REPO_ROOT / "workspace" / "DANN" / "scripts"


def _find_dann_ckpt(dann_path: Path) -> Path:
    pts = sorted(dann_path.glob("*.pt"))
    if not pts:
        raise SystemExit(
            f"No *.pt checkpoint under {dann_path}. Train DANN first or pass --dann-ckpt explicitly."
        )
    preferred = [p for p in pts if "best" in p.name.lower() or "dann" in p.name.lower()]
    return preferred[0] if preferred else pts[0]


def _find_feature_columns_json(dann_path: Path) -> Path:
    for name in ("source_train.feature_columns.json", "source_val.feature_columns.json"):
        p = dann_path / name
        if p.is_file():
            return p
    raise SystemExit(f"Missing source_train/source_val.feature_columns.json under {dann_path}")


def _resolve_ssl_embedding_parquet(ssl_embed_path: Path, validator_parquet: Path) -> Path:
    emb_dir = ssl_embed_path / "embeddings"
    cand = emb_dir / validator_parquet.name
    if cand.is_file():
        return cand
    fallback = emb_dir / "validator.parquet"
    if fallback.is_file():
        return fallback
    raise SystemExit(
        f"Missing SSL embedding parquet for validator. Expected:\n"
        f"  {cand}\n"
        f"or {fallback}\n"
        f"Re-run ssl ablation with validator included in export_embeddings (see run_ssl_lgbm_ablation.sh), "
        f"or pass --ssl-embeddings-parquet PATH."
    )


def _band(p: float, t_high: float, t_low: float, u_lo: float, u_hi: float) -> str:
    if p >= t_high:
        return "high_bot"
    if p <= t_low:
        return "high_human"
    if u_lo <= p <= u_hi:
        return "uncertain"
    return "medium"


def _run_infer_dann(*, ckpt: Path, export_npz: Path, out_npz: Path, device: str | None) -> None:
    cmd_inf = [
        sys.executable,
        str(DANN_SCRIPTS / "infer_dann.py"),
        "--ckpt",
        str(ckpt),
        "--npz",
        str(export_npz),
        "--out-npz",
        str(out_npz),
    ]
    if device:
        cmd_inf.extend(["--device", device])
    subprocess.run(cmd_inf, check=True, cwd=str(REPO_ROOT))


def _load_p_bot(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)
    return np.asarray(z["p_bot"], dtype=np.float64).reshape(-1)


def _weight(band: str, agreement: float, w_high: float, w_med: float, w_unc: float) -> float:
    if band == "high_bot" or band == "high_human":
        base = w_high
    elif band == "uncertain":
        base = w_unc
    else:
        base = w_med
    return float(max(0.0, min(1.0, base * agreement)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teacher pseudo-labels on validator parquet (DANN + optional SSL-LGBM, or DANN-only)."
    )
    p.add_argument(
        "--dann-path",
        type=Path,
        default=REPO_ROOT / "workspace" / "DANN" / "artifacts" / "v0001",
        help="DANN artifact dir (expects *.pt + source_*.feature_columns.json).",
    )
    p.add_argument(
        "--dann-ckpt",
        type=Path,
        default=None,
        help="Override DANN checkpoint .pt (default: first/best *.pt under --dann-path).",
    )
    p.add_argument(
        "--dann-ckpt-b",
        type=Path,
        default=None,
        help="Optional second DANN .pt (different seed/run). Enables two-checkpoint ensemble: "
        "p_dann=mean(p_a,p_b); agreement uses 1-|p_a-p_b| combined with margin or LGBM agreement.",
    )
    p.add_argument(
        "--dann-only",
        action="store_true",
        help="Use only DANN scores (no SSL embed / LGBM teacher). p_teacher=p_dann. "
        "With one ckpt: teacher_agreement = 2*|p_dann-0.5|. With --dann-ckpt-b: agreement combines "
        "seed agreement (1-|p_a-p_b|) and margin.",
    )
    p.add_argument(
        "--ssl-embed-path",
        type=Path,
        default=REPO_ROOT / "workspace" / "ssl_data" / "ssl_embed" / "artifacts" / "ssl_embed_v2",
        help="ssl_embed_v* run dir (expects lgbm_out/lgbm_b_classifier.joblib + embeddings/). Ignored with --dann-only.",
    )
    p.add_argument(
        "--ssl-embeddings-parquet",
        type=Path,
        default=None,
        help="Override path to embedded validator parquet (default: ssl-embed-path/embeddings/<validator basename>).",
    )
    p.add_argument(
        "--unlabeled-validator",
        type=Path,
        default=REPO_ROOT
        / "workspace"
        / "preprocess"
        / "statistical_test"
        / "explorer"
        / "feature_2"
        / "data"
        / "validator"
        / "validator.parquet",
        help="Raw validator parquet with DANN feature columns. Row order must match SSL embeddings only when not using --dann-only.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: workspace/teacher/artifacts/pseudo_teacher_<stem>).",
    )
    p.add_argument("--device", type=str, default=None, help="torch device for DANN (default: infer_dann default).")
    p.add_argument("--threshold-high-bot", type=float, default=0.95)
    p.add_argument("--threshold-high-human", type=float, default=0.05)
    p.add_argument("--uncertain-lo", type=float, default=0.40)
    p.add_argument("--uncertain-hi", type=float, default=0.60)
    p.add_argument("--w-high", type=float, default=1.0)
    p.add_argument("--w-medium", type=float, default=0.5)
    p.add_argument("--w-uncertain", type=float, default=0.2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dann_path = args.dann_path.expanduser().resolve()
    val_raw = args.unlabeled_validator.expanduser().resolve()
    dann_only = bool(args.dann_only)
    ssl_path = args.ssl_embed_path.expanduser().resolve() if not dann_only else None

    if not val_raw.is_file():
        raise SystemExit(f"Missing --unlabeled-validator: {val_raw}")
    if not dann_path.is_dir():
        raise SystemExit(f"Missing --dann-path directory: {dann_path}")
    if not dann_only and not ssl_path.is_dir():
        raise SystemExit(f"Missing --ssl-embed-path directory: {ssl_path}")

    ckpt = args.dann_ckpt.expanduser().resolve() if args.dann_ckpt else _find_dann_ckpt(dann_path)
    if not ckpt.is_file():
        raise SystemExit(f"DANN checkpoint not found: {ckpt}")
    ckpt_b_arg = args.dann_ckpt_b.expanduser().resolve() if args.dann_ckpt_b is not None else None
    feat_json = _find_feature_columns_json(dann_path)

    df_raw = pd.read_parquet(val_raw)
    emb_pq: Path | None = None
    model_path: Path | None = None
    df_emb: pd.DataFrame | None = None
    lgbm_cols: list[str] = []

    if not dann_only:
        assert ssl_path is not None
        lgbm_dir = ssl_path / "lgbm_out"
        model_path = lgbm_dir / "lgbm_b_classifier.joblib"
        lgbm_feat_path = lgbm_dir / "feature_cols.json"
        if not model_path.is_file():
            raise SystemExit(f"Missing LGBM model (train ssl_embed run first): {model_path}")
        if not lgbm_feat_path.is_file():
            raise SystemExit(f"Missing {lgbm_feat_path}")

        emb_pq = (
            args.ssl_embeddings_parquet.expanduser().resolve()
            if args.ssl_embeddings_parquet
            else _resolve_ssl_embedding_parquet(ssl_path, val_raw)
        )

        df_emb = pd.read_parquet(emb_pq)
        if len(df_raw) != len(df_emb):
            raise SystemExit(
                f"Row count mismatch: raw validator {len(df_raw)} vs SSL embeddings {len(df_emb)} "
                f"({emb_pq}). Regenerate embeddings from this exact parquet."
            )

        lgbm_meta = json.loads(lgbm_feat_path.read_text(encoding="utf-8"))
        lgbm_cols = list(lgbm_meta["feature_cols"])
        missing_l = [c for c in lgbm_cols if c not in df_emb.columns]
        if missing_l:
            raise SystemExit(f"SSL embedding parquet missing LGBM feature columns: {missing_l[:12]}")

    meta_fc = json.loads(feat_json.read_text(encoding="utf-8"))
    dann_cols = list(meta_fc["feature_columns"])
    missing = [c for c in dann_cols if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Raw validator missing DANN feature columns: {missing[:12]}")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else REPO_ROOT / "workspace" / "teacher" / "artifacts" / f"pseudo_teacher_{val_raw.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    export_npz = out_dir / "validator_dann_X.npz"
    dann_probs_npz_a = out_dir / "validator_p_dann_a.npz"
    dann_probs_npz_b = out_dir / "validator_p_dann_b.npz"

    # 1) Parquet -> NPZ (X) for DANN
    cmd_exp = [
        sys.executable,
        str(DANN_SCRIPTS / "export_parquet_to_target_npz.py"),
        "--parquet",
        str(val_raw),
        "--feature-columns-json",
        str(feat_json),
        "--out-npz",
        str(export_npz),
    ]
    subprocess.run(cmd_exp, check=True, cwd=str(REPO_ROOT))

    # 2) DANN inference (checkpoint A; optional B for seed ensemble)
    _run_infer_dann(ckpt=ckpt, export_npz=export_npz, out_npz=dann_probs_npz_a, device=args.device)
    p_a = _load_p_bot(dann_probs_npz_a)
    p_b = None
    if ckpt_b_arg is not None:
        if not ckpt_b_arg.is_file():
            raise SystemExit(f"--dann-ckpt-b not found: {ckpt_b_arg}")
        _run_infer_dann(ckpt=ckpt_b_arg, export_npz=export_npz, out_npz=dann_probs_npz_b, device=args.device)
        p_b = _load_p_bot(dann_probs_npz_b)
        if len(p_a) != len(p_b):
            raise SystemExit(f"DANN ensemble length mismatch: {len(p_a)} vs {len(p_b)}")
        p_dann = 0.5 * (p_a + p_b)
        seed_agree = 1.0 - np.abs(p_a - p_b)
    else:
        p_dann = p_a
        seed_agree = None

    margin = 2.0 * np.abs(p_dann - 0.5)

    if dann_only:
        # Single teacher path: no SSL/LGBM.
        p_lgbm = np.full_like(p_dann, np.nan, dtype=np.float64)
        p_teacher = p_dann.copy()
        if seed_agree is not None:
            agreement = seed_agree * margin
        else:
            agreement = margin
    else:
        # 3) LGBM on SSL features
        try:
            import joblib
        except ImportError as e:  # pragma: no cover
            raise SystemExit("joblib required for LGBM teacher") from e

        assert model_path is not None and df_emb is not None
        model = joblib.load(model_path)
        X_lgbm = df_emb.loc[:, lgbm_cols].to_numpy(dtype=np.float32, copy=True)
        if hasattr(model, "predict_proba"):
            p_lgbm = model.predict_proba(X_lgbm)[:, 1].astype(np.float64)
        else:
            raise SystemExit("Loaded model has no predict_proba")

        if len(p_dann) != len(p_lgbm):
            raise SystemExit(f"DANN/LGBM length mismatch: {len(p_dann)} vs {len(p_lgbm)}")

        p_teacher = 0.5 * (p_dann + p_lgbm)
        cross = 1.0 - np.abs(p_dann - p_lgbm)
        if seed_agree is not None:
            agreement = seed_agree * cross
        else:
            agreement = cross
    y_hat = (p_teacher >= 0.5).astype(np.int8)

    bands = [
        _band(float(p), args.threshold_high_bot, args.threshold_high_human, args.uncertain_lo, args.uncertain_hi)
        for p in p_teacher
    ]
    weights = [
        _weight(b, float(a), args.w_high, args.w_medium, args.w_uncertain)
        for b, a in zip(bands, agreement)
    ]

    out_df = df_raw.copy()
    out_df["p_dann"] = p_dann
    if p_b is not None:
        out_df["p_dann_a"] = p_a
        out_df["p_dann_b"] = p_b
    out_df["p_lgbm"] = p_lgbm
    out_df["p_teacher"] = p_teacher
    out_df["teacher_agreement"] = agreement
    out_df["y_hat"] = y_hat
    out_df["confidence_band"] = bands
    out_df["pseudo_weight"] = weights

    out_pq = out_dir / "pseudo_labeled_validator.parquet"
    out_df.to_parquet(out_pq, index=False)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_by": "build_teacher_pseudo_labels.py",
        "teacher_mode": "dann_only" if dann_only else "dann_ssl_lgbm",
        "dann_path": str(dann_path),
        "dann_ckpt": str(ckpt),
        "dann_ckpt_b": str(ckpt_b_arg) if ckpt_b_arg else None,
        "dann_ensemble": bool(ckpt_b_arg),
        "dann_feature_columns_json": str(feat_json),
        "unlabeled_validator_raw": str(val_raw),
        "n_rows": int(len(out_df)),
        "thresholds": {
            "high_bot_ge": args.threshold_high_bot,
            "high_human_le": args.threshold_high_human,
            "uncertain_lo": args.uncertain_lo,
            "uncertain_hi": args.uncertain_hi,
        },
        "weights": {"high": args.w_high, "medium": args.w_medium, "uncertain": args.w_uncertain},
        "outputs": {"parquet": str(out_pq)},
    }
    if not dann_only:
        manifest["ssl_embed_path"] = str(ssl_path)
        manifest["ssl_embeddings_parquet"] = str(emb_pq)
        manifest["lgbm_model"] = str(model_path)
        if ckpt_b_arg:
            manifest["agreement_note"] = (
                "Dual + 2 DANN ckpts: p_dann=mean(p_a,p_b); p_teacher=mean(p_dann,p_lgbm); "
                "teacher_agreement = (1-|p_a-p_b|) * (1-|p_dann-p_lgbm|)."
            )
    else:
        if ckpt_b_arg:
            manifest["agreement_note"] = (
                "dann_only + 2 ckpts: teacher_agreement = (1-|p_a-p_b|) * 2*|p_dann-0.5|; "
                "p_lgbm NaN; p_dann=mean(p_a,p_b)."
            )
        else:
            manifest["agreement_note"] = (
                "teacher_agreement = 2*|p_dann-0.5| (confidence margin); p_lgbm is NaN (unused)."
            )
    (out_dir / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "parquet": str(out_pq), "n_rows": len(out_df)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
