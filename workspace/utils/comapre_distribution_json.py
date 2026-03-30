#!/usr/bin/env python3
"""
Compare scalar distributions between poker hand JSON files (domain shift / EDA).

Expects each file to be a JSON array of hand dicts (metadata, players, actions, streets, outcome, optional label).

Memory: does **not** load the full JSON into RAM when using ``--max-hands`` (default for large files).
Uses ``ijson`` if installed (fast); otherwise streams the top-level array with the stdlib ``JSONDecoder``.

Plots (under ``workspace/utils/plots`` by default): boxplots (no outliers), overlaid ECDFs, normalized histograms,
and quantile–quantile vs the first dataset.

Example:
  python workspace/utils/comapre_distribution_json.py \\
    --input-json workspace/real_distribution/processed/merged_labeled.json \\
    --input-json hands_generator/human_hands/poker_hands_combined.json \\
    --max-hands 30000
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import ijson
except ImportError:
    ijson = None

_DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "plots"
_READ_CHUNK = 256 * 1024


def _permission_denied_outputs(out_dir: Path, err: PermissionError) -> str:
    target = getattr(err, "filename", None) or str(out_dir)
    return (
        f"Permission denied ({err}): {target}\n"
        f"Cannot write under {out_dir.resolve()}.\n"
        "Common cause: this folder or existing CSV/PNGs were created as root (e.g. Docker).\n"
        f"Fix ownership:  sudo chown -R \"$USER:$(id -gn)\" {out_dir}\n"
        "Or use a writable directory:  --out-dir /tmp/poker44_dist_plots"
    )


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    return None


def iter_json_array_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """
    Stream dict elements from a top-level JSON array without loading the whole file.
    Works for compact (single-line) arrays as long as each element is a complete JSON object.
    """
    decoder = json.JSONDecoder()
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        # Prime buffer until we see '['
        while True:
            chunk = f.read(_READ_CHUNK)
            if not chunk:
                raise ValueError(f"{path}: empty or invalid JSON")
            buf += chunk
            s = buf.lstrip()
            if s.startswith("\ufeff"):
                s = s[1:].lstrip()
            if not s.startswith("["):
                raise ValueError(f"{path}: streaming expects a top-level JSON array starting with '['")
            buf = s[1:].lstrip()
            break

        while True:
            buf = buf.lstrip()
            if not buf:
                chunk = f.read(_READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                continue
            if buf.startswith("]"):
                return
            while True:
                # Chunk boundaries can fall between `}` and `,`; the next read may start with `,`.
                # raw_decode rejects a leading comma, so strip it before each value.
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:].lstrip()
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    chunk = f.read(_READ_CHUNK)
                    if not chunk:
                        raise ValueError(f"{path}: truncated JSON (incomplete object)") from None
                    buf += chunk
                    continue
                buf = buf[idx:].lstrip()
                if buf.startswith(","):
                    buf = buf[1:].lstrip()
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}: expected each array element to be an object")
                yield obj
                break


def load_hands(
    path: Path,
    large_file_bytes: int,
    allow_full_load: bool,
) -> list[dict[str, Any]]:
    path = Path(path)
    size = path.stat().st_size

    if size >= large_file_bytes and not allow_full_load:
        raise SystemExit(
            f"{path.name} is ~{size / (1024**2):.1f} MiB. Refusing full load to avoid OOM. "
            f"Pass --max-hands N (streams via stdlib or ijson) or --allow-full-json-load."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "hands" in data:
            data = data["hands"]
        else:
            raise ValueError(f"{path}: expected a list or dict with 'hands', got keys {list(data)[:10]}")
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be a list of hands")
    return data


def hand_to_features(hand: dict[str, Any]) -> dict[str, float]:
    md = hand.get("metadata") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []
    outc = hand.get("outcome") or {}

    feats: dict[str, float] = {}

    for k in ("sb", "bb", "ante", "max_seats", "hero_seat", "button_seat"):
        v = _as_float(md.get(k))
        if v is not None:
            feats[f"meta_{k}"] = v

    n_players = len(players)
    feats["n_players"] = float(n_players)
    stacks = [
        _as_float(p.get("starting_stack"))
        for p in players
        if isinstance(p, dict) and _as_float(p.get("starting_stack")) is not None
    ]
    if stacks:
        feats["stack_mean"] = float(statistics.mean(stacks))
        feats["stack_min"] = float(min(stacks))
        feats["stack_max"] = float(max(stacks))
        if len(stacks) > 1:
            feats["stack_std"] = float(statistics.stdev(stacks))
        else:
            feats["stack_std"] = 0.0

    showed = sum(1 for p in players if isinstance(p, dict) and p.get("showed_hand"))
    feats["showed_frac"] = showed / max(n_players, 1)

    feats["n_actions"] = float(len(actions))
    feats["n_streets"] = float(len(streets))

    pots = [_as_float(a.get("pot_after")) for a in actions if isinstance(a, dict)]
    pots = [p for p in pots if p is not None]
    if pots:
        feats["pot_after_max"] = float(max(pots))
        feats["pot_after_last"] = float(pots[-1])

    nam = [
        _as_float(a.get("normalized_amount_bb"))
        for a in actions
        if isinstance(a, dict) and _as_float(a.get("normalized_amount_bb")) is not None
    ]
    if nam:
        feats["norm_bb_mean"] = float(statistics.mean(nam))
        feats["norm_bb_max"] = float(max(nam))
        if len(nam) > 1:
            feats["norm_bb_std"] = float(statistics.stdev(nam))
        else:
            feats["norm_bb_std"] = 0.0

    amounts = [_as_float(a.get("amount")) for a in actions if isinstance(a, dict)]
    amounts = [x for x in amounts if x is not None]
    if amounts:
        feats["amount_sum"] = float(sum(amounts))
        feats["amount_max"] = float(max(amounts))

    tp = _as_float(outc.get("total_pot"))
    if tp is not None:
        feats["outcome_total_pot"] = tp
    rk = _as_float(outc.get("rake"))
    if rk is not None:
        feats["outcome_rake"] = rk

    lab = hand.get("label")
    if lab is not None and not isinstance(lab, (dict, list)):
        fv = _as_float(lab)
        if fv is not None:
            feats["label"] = fv

    return feats


def stream_features_frame(path: Path, max_hands: int) -> pd.DataFrame:
    """Parse JSON in a streaming way and build a DataFrame without keeping raw hands."""
    path = Path(path)
    rows: list[dict[str, float]] = []
    if ijson is not None:
        with path.open("rb") as f:
            for i, item in enumerate(ijson.items(f, "item")):
                if not isinstance(item, dict):
                    raise ValueError(f"{path}: array elements must be objects")
                rows.append(hand_to_features(item))
                if i + 1 >= max_hands:
                    break
    else:
        for i, item in enumerate(iter_json_array_dicts(path)):
            rows.append(hand_to_features(item))
            if i + 1 >= max_hands:
                break
    return pd.DataFrame(rows)


def hands_to_frame(hands: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [hand_to_features(h) for h in hands]
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    desc = num.describe().T
    desc["missing_frac"] = num.isna().mean()
    return desc


def ks_table(
    df_a: pd.DataFrame, df_b: pd.DataFrame, min_count: int
) -> pd.DataFrame | None:
    if scipy_stats is None:
        return None
    cols_a = set(df_a.select_dtypes(include=[np.number]).columns)
    cols_b = set(df_b.select_dtypes(include=[np.number]).columns)
    common = sorted(cols_a & cols_b)
    rows = []
    for c in common:
        x = df_a[c].dropna().to_numpy(dtype=np.float64)
        y = df_b[c].dropna().to_numpy(dtype=np.float64)
        if len(x) < min_count or len(y) < min_count:
            continue
        stat, p = scipy_stats.ks_2samp(x, y, method="auto")
        rows.append({"feature": c, "ks_statistic": stat, "p_value": p, "n_a": len(x), "n_b": len(y)})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def _feature_order_for_plots(frames: dict[str, pd.DataFrame], labels: list[str], min_ks_n: int, top_k: int) -> list[str]:
    num_cols: set[str] = set()
    for df in frames.values():
        num_cols.update(df.select_dtypes(include=[np.number]).columns)
    common: set[str] | None = None
    for df in frames.values():
        c = set(df.select_dtypes(include=[np.number]).columns)
        common = c if common is None else (common & c)
    plot_candidates = sorted(common or num_cols)

    ranked: list[tuple[float, str]] = []
    if scipy_stats is not None and len(labels) >= 2:
        kt = ks_table(frames[labels[0]], frames[labels[1]], min_ks_n)
        if kt is not None and not kt.empty:
            for _, row in kt.iterrows():
                ranked.append((float(row["ks_statistic"]), str(row["feature"])))
    ranked.sort(reverse=True)
    order = [f for _, f in ranked] + [f for f in plot_candidates if f not in {x for _, x in ranked}]
    return order[:top_k]


def plot_boxplots(
    frames: dict[str, pd.DataFrame],
    out_path: Path,
    features: list[str],
    fig_cols: int = 3,
    dpi: int = 120,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots (pip install matplotlib)")
    n = len(features)
    if n == 0:
        return
    fig_rows = int(math.ceil(n / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.8 * fig_cols, 2.8 * fig_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(frames), 1)))

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        data = []
        labels = []
        for label, df in frames.items():
            s = df[feat].dropna().to_numpy(dtype=np.float32)
            if len(s) == 0:
                continue
            data.append(s)
            labels.append(f"{label}\n(n={len(s)})")
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], palette[: len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Distribution comparison — boxplots (no outliers)", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ecdfs(
    frames: dict[str, pd.DataFrame],
    out_path: Path,
    features: list[str],
    fig_cols: int = 3,
    dpi: int = 120,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots (pip install matplotlib)")
    n = len(features)
    if n == 0:
        return
    fig_rows = int(math.ceil(n / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.8 * fig_cols, 2.8 * fig_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        for label, df in frames.items():
            s = df[feat].dropna().to_numpy(dtype=np.float64)
            s.sort()
            if len(s) < 2:
                continue
            if hasattr(ax, "ecdf"):
                ax.ecdf(s, label=label, alpha=0.85)
            else:
                y = np.arange(1, len(s) + 1, dtype=np.float64) / len(s)
                ax.step(s, y, where="post", label=label, alpha=0.85)
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("F(x)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)

    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Empirical CDFs (domain shift)", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(
    frames: dict[str, pd.DataFrame],
    out_path: Path,
    features: list[str],
    fig_cols: int = 3,
    bins: int = 32,
    dpi: int = 120,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots (pip install matplotlib)")
    n = len(features)
    if n == 0:
        return
    fig_rows = int(math.ceil(n / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.8 * fig_cols, 2.6 * fig_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        for label, df in frames.items():
            s = df[feat].dropna().to_numpy(dtype=np.float32)
            if len(s) < 2:
                continue
            ax.hist(s, bins=bins, alpha=0.35, density=True, label=label, histtype="stepfilled", linewidth=0)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)

    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Normalized histograms", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_qq_vs_reference(
    frames: dict[str, pd.DataFrame],
    ref_label: str,
    out_path: Path,
    features: list[str],
    fig_cols: int = 3,
    n_quantiles: int = 100,
    dpi: int = 120,
) -> None:
    """QQ plot: each non-ref dataset vs reference quantiles (shape / shift)."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plots (pip install matplotlib)")
    if ref_label not in frames:
        return
    others = [k for k in frames if k != ref_label]
    if not others:
        return

    n = len(features)
    if n == 0:
        return
    fig_rows = int(math.ceil(n / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.8 * fig_cols, 2.8 * fig_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    ref_df = frames[ref_label]

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        r = ref_df[feat].dropna().to_numpy(dtype=np.float64)
        if len(r) < 10:
            ax.set_visible(False)
            continue
        probs = np.linspace(0.01, 0.99, min(n_quantiles, len(r)))
        q_ref = np.quantile(r, probs)
        for other in others:
            s = frames[other][feat].dropna().to_numpy(dtype=np.float64)
            if len(s) < 10:
                continue
            q_o = np.quantile(s, probs)
            ax.scatter(q_ref, q_o, s=4, alpha=0.5, label=f"{other} vs {ref_label}")
        lims = [np.nanmin(q_ref), np.nanmax(q_ref)]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="y=x")
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel(ref_label)
        ax.set_ylabel("other")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.25)
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"QQ vs reference «{ref_label}»", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _unique_labels(paths: list[Path]) -> list[str]:
    stems = [p.stem for p in paths]
    if len(stems) == len(set(stems)):
        return stems
    return [f"{p.stem}_{i}" for i, p in enumerate(paths)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-json",
        action="append",
        dest="input_json",
        required=True,
        help="Path to a JSON file (repeat for each dataset).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_PLOTS_DIR,
        help=f"Output directory (default: {_DEFAULT_PLOTS_DIR}).",
    )
    p.add_argument(
        "--max-hands",
        type=int,
        default=None,
        help="Max hands per file. If omitted and file < --large-file-mb, full load. If omitted and file is large, defaults to --default-max-hands.",
    )
    p.add_argument(
        "--default-max-hands",
        type=int,
        default=50_000,
        help="When file is large and --max-hands omitted, stream this many hands per file.",
    )
    p.add_argument(
        "--large-file-mb",
        type=float,
        default=16.0,
        help="Treat files >= this size as 'large' (stream / cap unless --allow-full-json-load).",
    )
    p.add_argument(
        "--allow-full-json-load",
        action="store_true",
        help="Allow json.load of large files (may OOM). Not recommended.",
    )
    p.add_argument(
        "--min-ks-n",
        type=int,
        default=30,
        help="Minimum non-null samples per side for KS in tables/plot ranking.",
    )
    p.add_argument(
        "--top-features",
        type=int,
        default=12,
        help="How many numeric features to include in multi-panel figures.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=120,
        help="Figure DPI (lower = faster, smaller files).",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=32,
        help="Histogram bin count.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print summaries / CSV; skip matplotlib.",
    )
    p.add_argument(
        "--qq-quantiles",
        type=int,
        default=100,
        help="Number of quantiles for QQ plots.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(p).resolve() for p in args.input_json]
    if len(paths) < 2:
        print("Need at least two --input-json paths.", file=sys.stderr)
        return 2

    large_bytes = int(args.large_file_mb * 1024 * 1024)
    labels = _unique_labels(paths)
    frames: dict[str, pd.DataFrame] = {}

    for path, label in zip(paths, labels):
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            return 2
        size = path.stat().st_size
        max_hands = args.max_hands
        if max_hands is None and size >= large_bytes and not args.allow_full_json_load:
            max_hands = args.default_max_hands
            print(
                f"{label}: large file (~{size / (1024**2):.1f} MiB) — streaming first {max_hands} hands "
                f"(set --max-hands or --allow-full-json-load to change).",
                file=sys.stderr,
            )

        if max_hands is not None:
            frames[label] = stream_features_frame(path, max_hands)
        else:
            hands = load_hands(path, large_bytes, args.allow_full_json_load)
            frames[label] = hands_to_frame(hands)
        print(f"{label}: {len(frames[label])} rows, {frames[label].shape[1]} features")

    for label, df in frames.items():
        print(f"\n=== Summary: {label} ===")
        print(numeric_summary(df).to_string())

    ordered_labels = [labels[i] for i in range(len(labels))]
    base = ordered_labels[0]
    if scipy_stats is not None and len(ordered_labels) >= 2:
        for other in ordered_labels[1:]:
            kt = ks_table(frames[base], frames[other], args.min_ks_n)
            if kt is not None and not kt.empty:
                print(f"\n=== KS two-sample ({base} vs {other}) — higher stat ⇒ more shift ===")
                print(kt.to_string(index=False))

    features = _feature_order_for_plots(frames, ordered_labels, args.min_ks_n, args.top_features)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for label, df in frames.items():
            numeric_summary(df).to_csv(out_dir / f"summary_{label}.csv")
        if len(ordered_labels) >= 2 and scipy_stats is not None:
            kt = ks_table(frames[ordered_labels[0]], frames[ordered_labels[1]], args.min_ks_n)
            if kt is not None and not kt.empty:
                kt.to_csv(out_dir / f"ks_{ordered_labels[0]}__vs__{ordered_labels[1]}.csv", index=False)

        if not args.no_plots and plt is not None and features:
            dpi = args.plot_dpi
            plot_boxplots(frames, out_dir / "boxplots.png", features, dpi=dpi)
            plot_ecdfs(frames, out_dir / "ecdf.png", features, dpi=dpi)
            plot_histograms(frames, out_dir / "histograms.png", features, bins=args.hist_bins, dpi=dpi)
            plot_qq_vs_reference(
                frames, base, out_dir / "qq_vs_first.png", features, n_quantiles=args.qq_quantiles, dpi=dpi
            )
            print(f"\nWrote plots and CSVs under {out_dir}")
        elif args.no_plots:
            print(f"\nWrote CSV summaries under {out_dir} (--no-plots)")
        elif plt is None:
            print("\nSkipped plots: matplotlib not installed.", file=sys.stderr)
    except PermissionError as e:
        print(_permission_denied_outputs(out_dir, e), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
