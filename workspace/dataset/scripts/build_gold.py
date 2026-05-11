#!/usr/bin/env python3
"""Build training parquet dataset from released gold benchmark JSON files.

This script enforces chunk-label correctness:
- one chunk (list[hand]) maps to exactly one label from groundTruth
- len(chunks) must equal len(groundTruth) for every payload block


python3 workspace/dataset/scripts/build_gold.py \
  --input-gold workspace/dataset/source/gold_dataset \
  --sample-features workspace/preprocess/statistical_test/explorer/feature_3/config/default/keep_features.txt \
  --output-dir workspace/preprocess/statistical_test/explorer/feature_3/data/gold/raw \
  --val-ratio 0.2 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_keep_features(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n]
    if not names:
        raise ValueError(f"feature file has no entries: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"feature file has duplicate feature names: {path}")
    return names


def _iter_gold_json(path: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            items = data.get("chunks")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item
                return
    raise ValueError(f"unrecognized gold JSON shape: {path}")


def _extract_chunk_label_pairs(item: Dict[str, Any], source: str) -> Iterable[Tuple[List[Dict[str, Any]], int]]:
    chunks = item.get("chunks")
    labels = item.get("groundTruth")
    if not isinstance(chunks, list) or not isinstance(labels, list):
        raise ValueError(f"{source}: missing list fields 'chunks'/'groundTruth'")
    if len(chunks) != len(labels):
        raise ValueError(
            f"{source}: chunk-label length mismatch len(chunks)={len(chunks)} len(groundTruth)={len(labels)}"
        )
    for idx, (chunk, label) in enumerate(zip(chunks, labels)):
        if not isinstance(chunk, list) or not chunk:
            continue
        try:
            label_i = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid label at index {idx}: {label!r}") from exc
        if label_i not in (0, 1):
            raise ValueError(f"{source}: label must be 0/1 at index {idx}, got {label_i}")
        yield chunk, label_i


def _stratified_split_indices(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not labels:
        return [], []
    by_label: Dict[int, List[int]] = {0: [], 1: []}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)

    rng = random.Random(seed)
    val_idx: List[int] = []
    train_idx: List[int] = []
    for y, indices in by_label.items():
        rng.shuffle(indices)
        if not indices:
            continue
        n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
        n_val = min(n_val, len(indices))
        part_val = indices[:n_val]
        part_train = indices[n_val:]
        val_idx.extend(part_val)
        train_idx.extend(part_train)
        if not part_train and part_val:
            moved = part_val.pop()
            train_idx.append(moved)
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build train/val parquet from gold benchmark JSON files.")
    ap.add_argument("--input-gold", type=Path, required=True, help="Directory containing downloaded gold *.json files.")
    ap.add_argument("--sample-features", type=Path, required=True, help="Feature name file (one feature per line).")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory where train.parquet and val.parquet are written.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    ap.add_argument("--log-every", type=int, default=5000, help="Progress logging cadence.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    input_gold = args.input_gold.expanduser().resolve()
    keep_file = args.sample_features.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_gold.is_dir():
        print(f"[build_gold] error: input dir missing: {input_gold}", file=sys.stderr)
        return 1
    if not keep_file.is_file():
        print(f"[build_gold] error: feature file missing: {keep_file}", file=sys.stderr)
        return 1
    if not (0.0 < args.val_ratio < 1.0):
        print("[build_gold] error: --val-ratio must be in (0, 1)", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT))
    from poker44.validator.chunk_features import aggregate_chunk_from_hands

    keep_features = _read_keep_features(keep_file)
    feature_columns = sorted(keep_features)
    keep_set = set(keep_features)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in input_gold.glob("*.json")
        if p.name != "releases.json" and not p.name.startswith("_")
    )
    if not files:
        print(f"[build_gold] error: no day json files under {input_gold}", file=sys.stderr)
        return 1

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    n_pairs = 0
    n_skipped_empty = 0

    for path in files:
        print(f"[build_gold] reading {path.name}")
        for item_idx, item in enumerate(_iter_gold_json(path)):
            src = f"{path.name}#item{item_idx}"
            for chunk, label in _extract_chunk_label_pairs(item, source=src):
                n_pairs += 1
                feat = aggregate_chunk_from_hands(chunk)
                if not feat:
                    n_skipped_empty += 1
                    continue
                missing = keep_set - set(feat.keys())
                if missing:
                    raise ValueError(
                        f"{src}: aggregator missing required features, e.g. {sorted(missing)[:10]}"
                    )
                row = {k: float(feat[k]) for k in feature_columns}
                rows.append(row)
                labels.append(label)
                if args.log_every and len(rows) % args.log_every == 0:
                    print(f"[build_gold] processed rows={len(rows)}")

    if not rows:
        print("[build_gold] error: no rows produced", file=sys.stderr)
        return 1

    train_idx, val_idx = _stratified_split_indices(labels, val_ratio=args.val_ratio, seed=args.seed)
    if not train_idx or not val_idx:
        print("[build_gold] error: split resulted in empty train or val set", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows, columns=feature_columns)
    df.insert(0, "label", pd.Series(labels, dtype="int64"))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    meta = {
        "input_gold": str(input_gold),
        "sample_features": str(keep_file),
        "feature_count": len(feature_columns),
        "total_pairs_seen": n_pairs,
        "rows_written": int(len(df)),
        "skipped_empty_agg": n_skipped_empty,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_label_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "val_label_counts": val_df["label"].value_counts().sort_index().to_dict(),
    }
    (output_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[build_gold] wrote train: {train_path}")
    print(f"[build_gold] wrote val:   {val_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
