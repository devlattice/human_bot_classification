#!/usr/bin/env python3
"""Shuffle and split poker_hands JSON (list of hands) into train/test (default 80:20)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "poker_hands_combined.json",
        help="Source JSON file (array of hand objects).",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=here / "poker_hands_train.json",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=here / "poker_hands_test.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of hands for training (rest goes to test). Default 0.8 (8:2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible shuffle.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in (0, 1)")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array, got {type(data).__name__}")

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n = len(data)
    n_train = int(round(n * args.train_ratio))
    train_hands = [data[i] for i in indices[:n_train]]
    test_hands = [data[i] for i in indices[n_train:]]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8") as f:
        json.dump(train_hands, f, ensure_ascii=False)
        f.write("\n")
    with open(args.test_out, "w", encoding="utf-8") as f:
        json.dump(test_hands, f, ensure_ascii=False)
        f.write("\n")

    print(
        f"Wrote {len(train_hands)} train, {len(test_hands)} test "
        f"(total {n}, ratio {args.train_ratio:.2f}) seed={args.seed}"
    )
    print(f"  {args.train_out}")
    print(f"  {args.test_out}")


if __name__ == "__main__":
    main()
