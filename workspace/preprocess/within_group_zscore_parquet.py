#!/usr/bin/env python3
"""
Within-group z-score for selected columns (column-level hybrid domain fix).

Fit on train only: for each discrete group (e.g. n_players_max), compute mean/std
per target column. Small groups fall back to global train stats.

Apply: same formula on any parquet using saved meta (validator, test, etc.).

Typical use after rb_* feature tables exist:
  fit  --data-dir .../train/concat/rb_B --out-dir .../train/concat/rb_B_wgz
  apply --transform-meta-in .../within_group_zscore_meta.json --in-parquet ... --out-parquet ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_split_df(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.parquet"
    if not path.is_file():
        raise SystemExit(f"Missing {path} (expected train.parquet and val.parquet under --data-dir)")
    return pd.read_parquet(path)


def _default_normalize_columns(columns: List[str], group_col: str) -> List[str]:
    """Shift-heavy stack/pot/action families; keep raw seat counts for export / nuisance."""
    prefixes = (
        "stack_",
        "mean_norm_bb_",
        "std_norm_bb_",
        "max_norm_bb_",
        "mean_pot_after_",
        "std_pot_after_",
        "raise_ratio_",
        "bet_ratio_",
        "call_ratio_",
        "check_ratio_",
        "fold_ratio_",
        "bet_minus_fold",
        "raise_minus_call",
        "raise_std_over_check",
        "late_minus_early",
        "pot_after_over_stack",
    )
    out: List[str] = []
    for c in columns:
        if c == "label" or c == group_col:
            continue
        if c.startswith("n_players") or c.startswith("n_streets") or c.startswith("n_actions"):
            continue
        if any(c.startswith(p) or c == p for p in prefixes):
            out.append(c)
    return sorted(set(out))


def _group_key_series(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        raise SystemExit(f"Missing group column {group_col!r}")
    g = pd.to_numeric(df[group_col], errors="coerce")
    g = np.round(g).astype("float64")
    g = g.clip(lower=2.0, upper=10.0)
    return g.astype("int64")


def _fit_stats(
    train_df: pd.DataFrame,
    group_col: str,
    cols: List[str],
    min_group_rows: int,
) -> Dict[str, Any]:
    gk = _group_key_series(train_df, group_col)
    per_group: Dict[str, Dict[str, Dict[str, float]]] = {}
    global_stats: Dict[str, Dict[str, float]] = {}

    for c in cols:
        s = pd.to_numeric(train_df[c], errors="coerce").astype(float)
        finite = s[np.isfinite(s)]
        if finite.empty:
            global_stats[c] = {"mean": 0.0, "std": 1.0}
            continue
        mu = float(finite.mean())
        sig = float(finite.std())
        if sig < 1e-12:
            sig = 1.0
        global_stats[c] = {"mean": mu, "std": sig}

    for gv, sub in train_df.groupby(gk):
        if len(sub) < int(min_group_rows):
            continue
        key = str(int(gv))
        per_group[key] = {}
        for c in cols:
            s = pd.to_numeric(sub[c], errors="coerce").astype(float)
            finite = s[np.isfinite(s)]
            if finite.empty:
                per_group[key][c] = dict(global_stats[c])
                continue
            mu = float(finite.mean())
            sig = float(finite.std())
            if sig < 1e-12:
                sig = 1.0
            per_group[key][c] = {"mean": mu, "std": sig}

    return {
        "version": 1,
        "group_col": group_col,
        "normalize_cols": cols,
        "min_group_rows": int(min_group_rows),
        "global": global_stats,
        "per_group": per_group,
    }


def _apply_df(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    group_col = str(meta["group_col"])
    cols = list(meta["normalize_cols"])
    global_stats = meta["global"]
    per_group: Dict[str, Any] = meta.get("per_group", {})

    out = df.copy()
    gk = _group_key_series(out, group_col)

    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        new_vals = np.full(len(out), np.nan, dtype=np.float64)
        for i in range(len(out)):
            key = str(int(gk.iloc[i]))
            st = per_group.get(key, {}).get(c) if isinstance(per_group.get(key), dict) else None
            if st is None:
                st = global_stats.get(c, {"mean": 0.0, "std": 1.0})
            mu, sig = float(st["mean"]), float(st["std"])
            if sig < 1e-12:
                sig = 1.0
            v = float(s.iloc[i])
            if not np.isfinite(v):
                new_vals[i] = v
            else:
                new_vals[i] = (v - mu) / sig
        out[c] = new_vals.astype(np.float64)
    return out


def _cmd_fit(args: argparse.Namespace) -> None:
    d = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split_df(d, "train")
    val_df = _load_split_df(d, "val")
    group_col = str(args.group_col)

    if args.normalize_cols_file:
        raw = Path(args.normalize_cols_file).read_text(encoding="utf-8").splitlines()
        cols = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]
        missing = [c for c in cols if c not in train_df.columns]
        if missing:
            raise SystemExit(f"--normalize-cols-file: missing columns: {missing[:12]}")
    else:
        cols = _default_normalize_columns(list(train_df.columns), group_col)

    meta = _fit_stats(train_df, group_col, cols, int(args.min_group_rows))
    meta["source_data_dir"] = str(d)

    train_out = _apply_df(train_df, meta)
    val_out = _apply_df(val_df, meta)

    meta_path = out_dir / "within_group_zscore_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    train_out.to_parquet(out_dir / "train.parquet", index=False)
    val_out.to_parquet(out_dir / "val.parquet", index=False)
    print(f"[within_group_zscore] wrote {out_dir / 'train.parquet'} {out_dir / 'val.parquet'}")
    print(f"[within_group_zscore] meta={meta_path}  n_cols={len(cols)}  groups_fit={len(meta['per_group'])}")


def _cmd_apply(args: argparse.Namespace) -> None:
    meta_path = Path(args.transform_meta_in).expanduser().resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inp = Path(args.in_parquet).expanduser().resolve()
    outp = Path(args.out_parquet).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing input: {inp}")
    df = pd.read_parquet(inp)
    out = _apply_df(df, meta)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outp, index=False)
    print(f"[within_group_zscore] {inp} -> {outp} rows={len(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Within-group z-score for parquets (hybrid column fix).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit on train, transform train+val, write meta + parquets")
    p_fit.add_argument("--data-dir", type=Path, required=True)
    p_fit.add_argument("--out-dir", type=Path, required=True)
    p_fit.add_argument("--group-col", type=str, default="n_players_max")
    p_fit.add_argument("--min-group-rows", type=int, default=30)
    p_fit.add_argument(
        "--normalize-cols-file",
        type=Path,
        default=None,
        help="Optional: one feature name per line; default uses stack/pot/action prefixes",
    )
    p_fit.set_defaults(func=_cmd_fit)

    p_ap = sub.add_parser("apply", help="Apply saved meta to one parquet")
    p_ap.add_argument("--transform-meta-in", type=Path, required=True)
    p_ap.add_argument("--in-parquet", type=Path, required=True)
    p_ap.add_argument("--out-parquet", type=Path, required=True)
    p_ap.set_defaults(func=_cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
