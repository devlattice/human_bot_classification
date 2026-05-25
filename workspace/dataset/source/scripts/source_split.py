#!/usr/bin/env python3
"""Shuffle and split poker hands into train/test (default 80:20).

Input may be a JSON array (.json / .json.gz) or JSON Lines (.jsonl / .jsonl.gz).
Output format follows the train/test path suffixes (same options).

python workspace/dataset/source/scripts/source_split.py \
  --input workspace/dataset/source/data/poker_hands_combined.json \
  --train-out workspace/dataset/source/data/poker_hands_train.json \
  --test-out workspace/dataset/source/data/poker_hands_test.json \
  --train-ratio 0.8 \
  --seed 42 \
  --progress-every 100000

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


class _AtomicStreamWriter:
    """Atomic streaming writer for json/jsonl (.gz supported)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.kind = _io_kind(self.path)
        self._tmp_path: Path | None = None
        self._fh = None
        self._count = 0
        self._first = True

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self) -> "_AtomicStreamWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suf = _suffix_for_tmp(self.path)
        fd, tmp = tempfile.mkstemp(
            suffix=suf,
            prefix=".tmp_split_",
            dir=str(self.path.parent),
        )
        os.close(fd)
        self._tmp_path = Path(tmp)
        if self.kind.endswith("_gz"):
            self._fh = gzip.open(self._tmp_path, "wt", encoding="utf-8")
        else:
            self._fh = self._tmp_path.open("w", encoding="utf-8")
        if self.kind in ("json_array", "json_array_gz"):
            self._fh.write("[")
        return self

    def write(self, row: Any) -> None:
        assert self._fh is not None
        payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if self.kind in ("jsonl", "jsonl_gz"):
            self._fh.write(payload + "\n")
        else:
            if not self._first:
                self._fh.write(",")
            self._fh.write(payload)
            self._first = False
        self._count += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._fh is not None and self.kind in ("json_array", "json_array_gz"):
                self._fh.write("]\n")
            if self._fh is not None:
                self._fh.close()
            if exc_type is None and self._tmp_path is not None:
                os.replace(self._tmp_path, self.path)
            elif self._tmp_path is not None and self._tmp_path.exists():
                self._tmp_path.unlink()
        finally:
            self._fh = None
            self._tmp_path = None


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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Progress print cadence in input rows for streaming mode (0 disables).",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise SystemExit("--train-ratio must be in (0, 1)")

    rng = random.Random(args.seed)
    input_kind = _io_kind(args.input)
    # CPU/RAM-friendly path for jsonl input: no full in-memory materialization.
    if input_kind in ("jsonl", "jsonl_gz"):
        if not args.input.is_file():
            raise SystemExit(f"Input not found: {args.input.resolve()}")
        seen = 0
        with _open_read(args.input) as src, _AtomicStreamWriter(
            args.train_out
        ) as train_w, _AtomicStreamWriter(args.test_out) as test_w:
            for row in _iter_jsonl_lines(src):
                seen += 1
                if rng.random() < float(args.train_ratio):
                    train_w.write(row)
                else:
                    test_w.write(row)
                pe = int(args.progress_every)
                if pe > 0 and seen % pe == 0:
                    print(
                        f"[stream-split] seen={seen} train={train_w.count} test={test_w.count}",
                        flush=True,
                    )
        n_train = train_w.count
        n_test = test_w.count
        n = n_train + n_test
    else:
        # Fallback for JSON array input: existing in-memory shuffle split.
        data = _load_hands(args.input)
        indices = list(range(len(data)))
        rng.shuffle(indices)

        n = len(data)
        n_train = int(round(n * args.train_ratio))
        train_hands = [data[i] for i in indices[:n_train]]
        test_hands = [data[i] for i in indices[n_train:]]
        n_test = len(test_hands)

        _atomic_write_hands(args.train_out, train_hands)
        _atomic_write_hands(args.test_out, test_hands)

    print(
        f"Wrote {n_train} train, {n_test} test "
        f"(total {n}, ratio {args.train_ratio:.2f}) seed={args.seed}"
    )
    print(f"  {args.train_out}")
    print(f"  {args.test_out}")


if __name__ == "__main__":
    main()
