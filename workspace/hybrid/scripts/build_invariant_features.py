"""Build domain-invariant features from existing parquet files.

Transforms magnitude features (pot, bet_size, norm_bb) into
self-referencing ratios that don't depend on blind structure,
table size, or era.

Reads each parquet from dataset/train/ and dataset/test/,
adds new *_inv columns, saves back.

Usage:
    python workspace/hybrid/scripts/build_invariant_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "train"
TEST_DIR = REPO_ROOT / "workspace" / "hybrid" / "dataset" / "test"


def add_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-invariant versions of magnitude features.

    Principle: replace absolute values with ratios/coefficients
    that capture the PATTERN without depending on SCALE.

    Example:
        Gold human:   mean_pot=12.5, std_pot=4.2  → CV=0.336
        Zenodo human: mean_pot=45.2, std_pot=15.1  → CV=0.334
        Same behavior, different scale → same invariant value.
    """
    df = df.copy()
    eps = 1e-8

    # ── Pot features: normalize by mean_pot_after_mean ──
    pot_base = df["mean_pot_after_mean"].clip(lower=eps)

    # Coefficient of variation: how variable are pot sizes?
    if "mean_pot_after_std" in df.columns:
        df["pot_cv_inv"] = df["mean_pot_after_std"] / pot_base

    # Relative spread: how different are p90 vs p10?
    if "mean_pot_after_p90" in df.columns and "mean_pot_after_p10" in df.columns:
        df["pot_spread_inv"] = (df["mean_pot_after_p90"] - df["mean_pot_after_p10"]) / pot_base

    # Relative median: where is the median relative to range?
    if "mean_pot_after_p50" in df.columns and "mean_pot_after_p90" in df.columns and "mean_pot_after_p10" in df.columns:
        pot_range = (df["mean_pot_after_p90"] - df["mean_pot_after_p10"]).clip(lower=eps)
        df["pot_median_position_inv"] = (df["mean_pot_after_p50"] - df["mean_pot_after_p10"]) / pot_range

    # Pot growth relative to pot size
    if "pot_growth_mean" in df.columns:
        df["pot_growth_relative_inv"] = df["pot_growth_mean"] / pot_base
    if "pot_growth_std" in df.columns:
        df["pot_growth_cv_inv"] = df["pot_growth_std"] / pot_base

    # Std pot after: coefficient of variation
    if "std_pot_after_std" in df.columns:
        df["std_pot_cv_inv"] = df["std_pot_after_std"] / pot_base

    # pot_after_over_stack is already a ratio but still shifts.
    # Normalize by its own mean to capture pattern.
    # Actually keep as-is since it's conceptually a ratio already.

    # ── Bet size features: normalize by mean bet size ──
    bet_base = df.get("bet_size_mean_mean")
    if bet_base is not None:
        bet_base = bet_base.clip(lower=eps)

        if "bet_size_mean_std" in df.columns:
            df["bet_size_cv_inv"] = df["bet_size_mean_std"] / bet_base

        if "bet_size_mean_p90" in df.columns:
            df["bet_size_p90_ratio_inv"] = df["bet_size_mean_p90"] / bet_base

        if "bet_size_mean_p50" in df.columns:
            df["bet_size_p50_ratio_inv"] = df["bet_size_mean_p50"] / bet_base

        if "bet_size_mean_max" in df.columns:
            df["bet_size_max_ratio_inv"] = df["bet_size_mean_max"] / bet_base

        if "bet_size_max_std" in df.columns:
            df["bet_size_max_cv_inv"] = df["bet_size_max_std"] / bet_base

        if "bet_size_std_max" in df.columns:
            df["bet_size_std_max_ratio_inv"] = df["bet_size_std_max"] / bet_base

    # ── Norm BB features: normalize by spread ──
    if "mean_norm_bb_p90" in df.columns and "std_norm_bb_std" in df.columns:
        bb_base = df["mean_norm_bb_p90"].clip(lower=eps)
        df["norm_bb_cv_inv"] = df["std_norm_bb_std"] / bb_base

    if "max_norm_bb_std" in df.columns and "mean_norm_bb_p90" in df.columns:
        bb_base = df["mean_norm_bb_p90"].clip(lower=eps)
        df["max_norm_bb_cv_inv"] = df["max_norm_bb_std"] / bb_base

    # ── Cross-feature ratios ──
    # Bet size relative to pot (how much of pot does player bet?)
    if "bet_size_mean_mean" in df.columns and "mean_pot_after_mean" in df.columns:
        df["bet_to_pot_ratio_inv"] = df["bet_size_mean_mean"] / pot_base

    # Growth volatility: is pot growth consistent?
    if "pot_growth_mean" in df.columns and "pot_growth_std" in df.columns:
        pg_mean = df["pot_growth_mean"].clip(lower=eps)
        df["pot_growth_consistency_inv"] = df["pot_growth_std"] / pg_mean

    # Replace NaN/inf with 0
    inv_cols = [c for c in df.columns if c.endswith("_inv")]
    for c in inv_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def process_directory(dir_path: Path):
    """Process all parquet files in a directory."""
    parquet_files = sorted(dir_path.glob("*.parquet"))
    if not parquet_files:
        print(f"  No parquet files in {dir_path}")
        return

    for p in parquet_files:
        df = pd.read_parquet(p)
        n_before = len(df.columns)
        df = add_invariant_features(df)
        n_after = len(df.columns)
        n_new = n_after - n_before
        df.to_parquet(p, index=False)
        inv_cols = [c for c in df.columns if c.endswith("_inv")]
        print("  {:<45s} {} rows, +{} inv features (total {})".format(
            p.name, len(df), n_new, len(inv_cols)))


def main():
    print("Building domain-invariant features")
    print("=" * 60)

    print("\nTrain data:")
    process_directory(TRAIN_DIR)

    print("\nTest data:")
    process_directory(TEST_DIR)

    # Show the new features
    sample = pd.read_parquet(next(TRAIN_DIR.glob("*.parquet")))
    inv_cols = sorted([c for c in sample.columns if c.endswith("_inv")])
    print("\nNew invariant features ({}):\n".format(len(inv_cols)))
    for c in inv_cols:
        print("  {}".format(c))

    print("\nDone.")


if __name__ == "__main__":
    main()
