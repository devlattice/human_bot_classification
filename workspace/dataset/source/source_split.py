#!/usr/bin/env python3
"""Shuffle and split poker hands into train/test (default 80:20).

Input may be a JSON array (.json / .json.gz) or JSON Lines (.jsonl / .jsonl.gz).
Output format follows the train/test path suffixes (same options).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator, List


def _io_kind(path: Path) -> str:
    """Return json_array | json_array_gz | jsonl | jsonl_gz."""
    n = path.name.lower()
    if n.endswith(".jsonl.gz"):
        return "jsonl_gz"
    if n.endswith(".jsonl"):
        return "jsonl"
    if n.endswith(".json.gz"):
        return "json_array_gz"
    if n.endswith(".json"):
        return "json_array"
    raise SystemExit(
        f"Unsupported output/input suffix (use .json, .json.gz, .jsonl, .jsonl.gz): {path}"
    )


def _open_read(path: Path):
    kind = _io_kind(path)
    if kind.endswith("_gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _iter_jsonl_lines(f) -> Iterator[Any]:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"JSONL parse error at line {line_no}: {e}") from e


def _load_hands(path: Path) -> List[Any]:
    if not path.is_file():
        raise SystemExit(f"Input not found: {path.resolve()}")
    kind = _io_kind(path)
    with _open_read(path) as f:
        if kind.startswith("jsonl"):
            return list(_iter_jsonl_lines(f))
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array, got {type(data).__name__}")
    return data


def _suffix_for_tmp(path: Path) -> str:
    n = path.name.lower()
    if n.endswith(".jsonl.gz"):
        return ".jsonl.gz"
    if n.endswith(".jsonl"):
        return ".jsonl"
    if n.endswith(".json.gz"):
        return ".json.gz"
    return ".json"


def _atomic_write_hands(path: Path, rows: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = _suffix_for_tmp(path)
    fd, tmp = tempfile.mkstemp(
        suffix=suf,
        prefix=".tmp_split_",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    kind = _io_kind(path)
    try:
        if kind == "jsonl_gz":
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        elif kind == "jsonl":
            with tmp_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        elif kind == "json_array_gz":
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
        else:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "poker_hands_combined.json",
        help="Source: JSON array (.json/.json.gz) or JSON Lines (.jsonl/.jsonl.gz).",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=here / "poker_hands_train.json",
        help="Train output (.json, .json.gz, .jsonl, .jsonl.gz).",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=here / "poker_hands_test.json",
        help="Test output (.json, .json.gz, .jsonl, .jsonl.gz).",
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

    data = _load_hands(args.input)

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n = len(data)
    n_train = int(round(n * args.train_ratio))
    train_hands = [data[i] for i in indices[:n_train]]
    test_hands = [data[i] for i in indices[n_train:]]

    _atomic_write_hands(args.train_out, train_hands)
    _atomic_write_hands(args.test_out, test_hands)

    print(
        f"Wrote {len(train_hands)} train, {len(test_hands)} test "
        f"(total {n}, ratio {args.train_ratio:.2f}) seed={args.seed}"
    )
    print(f"  {args.train_out}")
    print(f"  {args.test_out}")


if __name__ == "__main__":
    main()
