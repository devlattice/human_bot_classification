"""KS (Kolmogorov-Smirnov) test analysis on selected features.

Produces three analyses:
  1. Human vs Bot separation (gold data) — feature discriminative power
  2. Human domain stability (gold vs zenodo vs public vs wsop) — FPR risk
  3. Bot domain stability (gold bot vs acpc bot vs generated bot) — generalization

Outputs:
  workspace/hybrid/KS_test/
    ks_human_vs_bot.png        — bar chart of KS stats per feature
    ks_human_domain_shift.png  — human cross-domain KS
    ks_bot_domain_shift.png    — bot cross-domain KS
    ks_results.csv             — full numeric results
    ks_summary.txt             — text summary

Usage:
    python workspace/hybrid/scripts/ks_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_DIR = REPO_ROOT / "workspace" / "hybrid" / "model_bundle"
TRAIN_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train"
TEST_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test"
OUTPUT_DIR = REPO_ROOT / "workspace" / "hybrid" / "KS_test"


def load_feature_cols():
    fc_path = BUNDLE_DIR / "feature_cols.json"
    return json.loads(fc_path.read_text())["feature_cols"]


def safe_load(path: Path, feature_cols: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARN: {path.name} missing {len(missing)} features, skipping")
        return None
    return df


def ks_between(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return KS statistic and p-value, handling edge cases."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    stat, pval = ks_2samp(a, b)
    return float(stat), float(pval)


def plot_ks_bars(results: list[dict], title: str, out_path: Path, color_key: str = "ks_stat"):
    """Horizontal bar chart sorted by KS statistic."""
    results = sorted(results, key=lambda r: r[color_key], reverse=True)
    names = [r["feature"] for r in results]
    stats = [r[color_key] for r in results]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.28)))

    colors = []
    for s in stats:
        if s >= 0.5:
            colors.append("#2ecc71")  # green = strong separation
        elif s >= 0.3:
            colors.append("#f39c12")  # orange = moderate
        elif s >= 0.1:
            colors.append("#e74c3c")  # red = weak
        else:
            colors.append("#95a5a6")  # gray = negligible

    ax.barh(range(len(names)), stats, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("KS Statistic")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0.5, color="#2ecc71", linestyle="--", alpha=0.5, label="Strong (≥0.5)")
    ax.axvline(x=0.3, color="#f39c12", linestyle="--", alpha=0.5, label="Moderate (≥0.3)")
    ax.axvline(x=0.1, color="#e74c3c", linestyle="--", alpha=0.5, label="Weak (≥0.1)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_domain_shift(all_results: dict[str, list[dict]], title: str, out_path: Path):
    """Grouped bar chart: KS stats across domain pairs per feature."""
    pair_names = list(all_results.keys())
    if not pair_names:
        return
    features = [r["feature"] for r in all_results[pair_names[0]]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.3)))
    bar_height = 0.8 / len(pair_names)
    cmap = plt.cm.Set2

    for pi, pname in enumerate(pair_names):
        stats = {r["feature"]: r["ks_stat"] for r in all_results[pname]}
        vals = [stats.get(f, 0) for f in features]
        positions = [i + pi * bar_height for i in range(len(features))]
        ax.barh(positions, vals, height=bar_height, label=pname,
                color=cmap(pi / max(len(pair_names) - 1, 1)), edgecolor="white", linewidth=0.3)

    ax.set_yticks([i + bar_height * (len(pair_names) - 1) / 2 for i in range(len(features))])
    ax.set_yticklabels(features, fontsize=7)
    ax.set_xlabel("KS Statistic (lower = more stable across domains)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.4, label="Risk threshold (0.2)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = load_feature_cols()
    print(f"Features: {len(feature_cols)}")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    # ─── Load datasets ───
    gold = safe_load(TRAIN_DIR / "gold_features.parquet", feature_cols)
    zenodo = safe_load(TRAIN_DIR / "zenodo_features.parquet", feature_cols)
    public = safe_load(TRAIN_DIR / "public_features.parquet", feature_cols)
    gen_bot = safe_load(TRAIN_DIR / "generated_bot_features.parquet", feature_cols)
    acpc_train = safe_load(TRAIN_DIR / "acpc_bot_features.parquet", feature_cols)
    full_spec = safe_load(TRAIN_DIR / "full_spectrum_bot_features.parquet", feature_cols)

    zen_test = safe_load(TEST_DIR / "zenodo_test_features.parquet", feature_cols)
    pub_test = safe_load(TEST_DIR / "public_test_features.parquet", feature_cols)
    acpc_test = safe_load(TEST_DIR / "acpc_bot_test_features.parquet", feature_cols)
    wsop = safe_load(TEST_DIR / "wsop_stress_features.parquet", feature_cols)

    all_csv_rows = []

    # ═══════════════════════════════════════════════════════════
    # Analysis 1: Human vs Bot (gold data)
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Analysis 1: Human vs Bot separation (gold)")
    print("=" * 60)

    if gold is not None:
        gold_h = gold[gold["label"] == 0]
        gold_b = gold[gold["label"] == 1]
        hvb_results = []
        for f in feature_cols:
            ks, pv = ks_between(gold_h[f].values, gold_b[f].values)
            hvb_results.append({"feature": f, "ks_stat": ks, "p_value": pv})
            all_csv_rows.append({"analysis": "human_vs_bot_gold", "feature": f, "ks_stat": ks, "p_value": pv})

        hvb_results.sort(key=lambda r: r["ks_stat"], reverse=True)
        strong = sum(1 for r in hvb_results if r["ks_stat"] >= 0.5)
        moderate = sum(1 for r in hvb_results if 0.3 <= r["ks_stat"] < 0.5)
        weak = sum(1 for r in hvb_results if r["ks_stat"] < 0.3)
        print(f"  Strong (≥0.5): {strong} | Moderate (0.3-0.5): {moderate} | Weak (<0.3): {weak}")
        top5 = ", ".join(f"{r['feature']}({r['ks_stat']:.3f})" for r in hvb_results[:5])
        print(f"  Top 5: {top5}")

        plot_ks_bars(hvb_results,
                     f"Human vs Bot KS Distance (gold, {len(gold_h)}h vs {len(gold_b)}b)",
                     OUTPUT_DIR / "ks_human_vs_bot.png")

    # ═══════════════════════════════════════════════════════════
    # Analysis 2: Human domain shift
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Analysis 2: Human domain stability")
    print("=" * 60)

    human_sources = {}
    if gold is not None:
        human_sources["gold_human"] = gold[gold["label"] == 0]
    if zenodo is not None:
        human_sources["zenodo_train"] = zenodo.sample(n=min(3000, len(zenodo)), random_state=42)
    if public is not None:
        human_sources["public_train"] = public
    if zen_test is not None:
        human_sources["zenodo_test"] = zen_test.sample(n=min(3000, len(zen_test)), random_state=42)
    if pub_test is not None:
        human_sources["public_test"] = pub_test
    if wsop is not None and len(wsop) > 0:
        human_sources["wsop_2023"] = wsop

    human_shift_results = {}
    hnames = list(human_sources.keys())
    for i in range(len(hnames)):
        for j in range(i + 1, len(hnames)):
            pair = f"{hnames[i]} vs {hnames[j]}"
            da = human_sources[hnames[i]]
            db = human_sources[hnames[j]]
            pair_res = []
            for f in feature_cols:
                ks, pv = ks_between(da[f].values, db[f].values)
                pair_res.append({"feature": f, "ks_stat": ks, "p_value": pv})
                all_csv_rows.append({"analysis": f"human_shift_{pair}", "feature": f, "ks_stat": ks, "p_value": pv})
            human_shift_results[pair] = sorted(pair_res, key=lambda r: r["ks_stat"], reverse=True)

    for pair, res in human_shift_results.items():
        mean_ks = np.mean([r["ks_stat"] for r in res])
        risky = sum(1 for r in res if r["ks_stat"] >= 0.3)
        print(f"  {pair}: mean_KS={mean_ks:.3f}, risky_features(≥0.3)={risky}")

    if human_shift_results:
        ref_pair = list(human_shift_results.keys())[0]
        feature_order = [r["feature"] for r in human_shift_results[ref_pair]]
        ordered_results = {}
        for pair, res in human_shift_results.items():
            ordered_results[pair] = sorted(res, key=lambda r: feature_order.index(r["feature"]))
        plot_domain_shift(ordered_results,
                          "Human Domain Shift (lower = more stable)",
                          OUTPUT_DIR / "ks_human_domain_shift.png")

    # ═══════════════════════════════════════════════════════════
    # Analysis 3: Bot domain shift
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Analysis 3: Bot domain stability")
    print("=" * 60)

    bot_sources = {}
    if gold is not None:
        gb = gold[gold["label"] == 1]
        if len(gb) > 0:
            bot_sources["gold_bot"] = gb
    if gen_bot is not None:
        bot_sources["generated"] = gen_bot.sample(n=min(3000, len(gen_bot)), random_state=42)
    if full_spec is not None:
        bot_sources["full_spectrum"] = full_spec.sample(n=min(3000, len(full_spec)), random_state=42)
    if acpc_train is not None:
        bot_sources["acpc_train"] = acpc_train
    if acpc_test is not None:
        bot_sources["acpc_test"] = acpc_test

    bot_shift_results = {}
    bnames = list(bot_sources.keys())
    for i in range(len(bnames)):
        for j in range(i + 1, len(bnames)):
            pair = f"{bnames[i]} vs {bnames[j]}"
            da = bot_sources[bnames[i]]
            db = bot_sources[bnames[j]]
            pair_res = []
            for f in feature_cols:
                ks, pv = ks_between(da[f].values, db[f].values)
                pair_res.append({"feature": f, "ks_stat": ks, "p_value": pv})
                all_csv_rows.append({"analysis": f"bot_shift_{pair}", "feature": f, "ks_stat": ks, "p_value": pv})
            bot_shift_results[pair] = sorted(pair_res, key=lambda r: r["ks_stat"], reverse=True)

    for pair, res in bot_shift_results.items():
        mean_ks = np.mean([r["ks_stat"] for r in res])
        risky = sum(1 for r in res if r["ks_stat"] >= 0.3)
        print(f"  {pair}: mean_KS={mean_ks:.3f}, risky_features(≥0.3)={risky}")

    if bot_shift_results:
        ref_pair = list(bot_shift_results.keys())[0]
        feature_order = [r["feature"] for r in bot_shift_results[ref_pair]]
        ordered_results = {}
        for pair, res in bot_shift_results.items():
            ordered_results[pair] = sorted(res, key=lambda r: feature_order.index(r["feature"]))
        plot_domain_shift(ordered_results,
                          "Bot Domain Shift (lower = more generalizable)",
                          OUTPUT_DIR / "ks_bot_domain_shift.png")

    # ═══════════════════════════════════════════════════════════
    # Save CSV + summary
    # ═══════════════════════════════════════════════════════════
    csv_df = pd.DataFrame(all_csv_rows)
    csv_path = OUTPUT_DIR / "ks_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path} ({len(csv_df)} rows)")

    # Summary text
    lines = []
    lines.append("=" * 70)
    lines.append("KS ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Features analyzed: {len(feature_cols)}")
    lines.append("")

    if gold is not None:
        lines.append("─── Human vs Bot (gold) ───")
        for r in sorted(hvb_results, key=lambda r: r["ks_stat"], reverse=True)[:10]:
            sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
            lines.append(f"  {r['feature']:<40s} KS={r['ks_stat']:.4f} {sig}")
        lines.append("")

    lines.append("─── Human domain shift (mean KS per pair) ───")
    for pair, res in human_shift_results.items():
        mean_ks = np.mean([r["ks_stat"] for r in res])
        lines.append(f"  {pair:<45s} mean_KS={mean_ks:.4f}")
    lines.append("")

    lines.append("─── Bot domain shift (mean KS per pair) ───")
    for pair, res in bot_shift_results.items():
        mean_ks = np.mean([r["ks_stat"] for r in res])
        lines.append(f"  {pair:<45s} mean_KS={mean_ks:.4f}")

    # Feature risk assessment
    lines.append("")
    lines.append("─── Feature risk assessment ───")
    lines.append("Features with HIGH human domain shift (>0.3 in any pair):")
    risky_feats = set()
    for pair, res in human_shift_results.items():
        for r in res:
            if r["ks_stat"] >= 0.3:
                risky_feats.add(r["feature"])
    if risky_feats:
        for f in sorted(risky_feats):
            lines.append(f"  ⚠ {f}")
    else:
        lines.append("  None — all features stable across human domains")

    summary_text = "\n".join(lines)
    summary_path = OUTPUT_DIR / "ks_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"  Saved: {summary_path}")

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
