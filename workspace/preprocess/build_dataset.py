#!/usr/bin/env python3
"""
Build LightGBM training data from Poker44-style labeled chunks.

Pipeline:
  1) Load labeled chunks (human vs bot) from the mixed dataset builder or random generator.
  2) Apply sanitize_hand_for_miner() to every hand (same view as live miners).
  3) Extract fixed numeric features per hand, aggregate to one row per chunk.
  4) Run data checks (train/serve parity, labels, NaN/inf, class balance).
  5) Optionally enforce **miner-servable** columns only (same schema as ``aggregate_chunk_from_hands``).
  6) Stratified train/val split → Parquet (or CSV).

Each Parquet **row** is one **chunk** (aggregated features over many hands), not one hand.
``chunk_count`` controls row count; human hands pulled from ``--human-json`` scale roughly as
``≈ (chunk_count * human_ratio) * mean(hands_per_chunk)``. Default ``training`` (120 chunks)
uses only a few thousand human hands per run even if the JSON has 100k+ hands — use
``--preset training-merged`` or a large ``--chunk-count`` to sample more.

**Sharded mode** (``--shard-size``): builds the dataset in independent shards (lower peak RAM,
resumable). Each shard runs ``build_mixed_labeled_chunks`` with a subset of ``chunk_count``,
writes ``out/shards/shard_XXXXX.parquet``, then concatenates all shards and applies **one**
global stratified train/val split. Shards are not identical to a single monolithic run with the
same seed (per-shard seeds/windows differ); use for large ``chunk_count`` when the process is
OOM-killed or too slow.

Run from repo root:
  pip install -r workspace/datasets/requirements-dataset.txt
  PYTHONPATH=. python workspace/_subnet_target/preprocess/build_dataset.py --help

Defaults target the subnet human+bot pipeline (train split JSON + unpreprocessed output under
``workspace/_subnet_target/dataset/``). For the shared workspace preprocessor with different
defaults, use ``workspace/datasets/_preprocess/preprocess_lightgbm.py``.

Merge existing shards only (no generation):
  PYTHONPATH=. python workspace/_subnet_target/preprocess/build_dataset.py \\
      --merge-only path/to/out --val-size 0.2 --seed 42

See workspace/docs/DATASET.md for context.

**Miner alignment:** use ``--miner-feature-schema strict`` (default) so Parquet has exactly the
features a live miner can compute from validator chunks. Extra columns (e.g. after manual merges)
cause the miner to zero-fill at inference. Use ``strip`` to drop extras, or ``off`` for legacy data.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Repo root: this file lives under workspace/_subnet_target/preprocess/
def _repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "poker44").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not locate repo root (no poker44/ directory above this file).")


REPO_ROOT = _repo_root()
_THIS_DIR = Path(__file__).resolve().parent
# Single-source human pool for subnet training (after source_split.py); override with --human-json.
SUBNET_DEFAULT_HUMAN_JSON = (
    REPO_ROOT / "workspace" / "_subnet_target" / "dataset" / "source" / "poker_hands_train.json"
)
SUBNET_DEFAULT_OUT = (
    REPO_ROOT
    / "workspace"
    / "_subnet_target"
    / "dataset"
    / "unpreprocessed"
    / "original_train"
)


def _rel_repo(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


for _p in (REPO_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dataset_presets import PRESETS, PresetName

from hands_generator.mixed_dataset_provider import (
    MixedDatasetConfig,
    build_mixed_labeled_chunks,
)
from mixed_dataset_compat import (
    all_bot_profile_modes,
    filter_mixed_dataset_config_kwargs,
)
from poker44.validator.chunk_features import aggregate_chunk_from_hands, miner_servable_feature_names
from poker44.validator.sanitization import sanitize_hand_for_miner

# Keep in sync with poker44.validator.sanitization._LEAKAGE_KEYS
MINER_LEAKAGE_KEYS: Set[str] = {
    "label",
    "label_flag",
    "is_bot",
    "bot_family_id",
    "bot_version",
}

EXPECTED_ACTION_SLOTS = 12

# Decorrelate per-shard MixedDatasetConfig.seed from the user base seed.
_SHARD_SEED_STRIDE = 100_003

MinerFeatureSchemaMode = Literal["strict", "strip", "off"]


class DataValidationError(Exception):
    """Raised when pre-training data checks fail."""


def _effective_chunk_params(
    preset: PresetName,
    *,
    chunk_count: Optional[int],
    min_hands: Optional[int],
    max_hands: Optional[int],
    human_ratio: Optional[float],
) -> Tuple[int, int, int, float]:
    base = PRESETS[preset]
    cc = int(chunk_count if chunk_count is not None else base["chunk_count"])
    if cc < 1:
        raise DataValidationError("chunk_count must be >= 1.")
    mn = int(min_hands if min_hands is not None else base["min_hands_per_chunk"])
    mx = int(max_hands if max_hands is not None else base["max_hands_per_chunk"])
    hr = float(human_ratio if human_ratio is not None else base["human_ratio"])
    return cc, mn, mx, hr


def _mkdir_output_dir(path: Path) -> None:
    """Create ``path`` (and parents); explain permission errors (common if tree is root-owned)."""
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        subnet_ds = REPO_ROOT / "workspace" / "_subnet_target" / "dataset"
        raise PermissionError(
            f"Cannot create output directory {p} ({e}). "
            "Use --out with a writable path. If this is under subnet defaults, try: "
            f'sudo chown -R "$USER:$USER" {subnet_ds}'
        ) from e


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise DataValidationError(f"Human JSON source not found: {p}")
    if p.suffix.lower() == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise DataValidationError(f"{p}: expected top-level JSON array.")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise DataValidationError(f"{p}: item {i} is not an object.")
        out.append(item)
    return out


def _write_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _build_balanced_human_pool(
    *,
    source_a: Path,
    source_b: Path,
    ratio_a: float,
    seed: int,
    out_path: Path,
    sample_with_replacement: bool,
) -> Tuple[Path, Dict[str, Any]]:
    if not (0.0 < ratio_a < 1.0):
        raise DataValidationError("--human-source-ratio-a must be in (0,1), e.g. 0.5")
    a_rows = _load_json_array(source_a)
    b_rows = _load_json_array(source_b)
    if not a_rows or not b_rows:
        raise DataValidationError("Both human sources must be non-empty.")

    rng = random.Random(int(seed))
    n_a = len(a_rows)
    n_b = len(b_rows)

    if sample_with_replacement:
        total = max(n_a, n_b) * 2
        take_a = int(round(total * ratio_a))
        take_b = total - take_a
        if take_a < 1 or take_b < 1:
            raise DataValidationError("Invalid ratio produced empty side with replacement.")
        a_sel = [a_rows[rng.randrange(n_a)] for _ in range(take_a)]
        b_sel = [b_rows[rng.randrange(n_b)] for _ in range(take_b)]
    else:
        # Max total that preserves desired ratio without replacement.
        total_by_a = n_a / ratio_a
        total_by_b = n_b / (1.0 - ratio_a)
        total = int(min(total_by_a, total_by_b))
        if total < 2:
            raise DataValidationError("Sources too small for requested ratio without replacement.")
        take_a = int(round(total * ratio_a))
        take_b = total - take_a
        take_a = max(1, min(take_a, n_a))
        take_b = max(1, min(take_b, n_b))

        a_idx = list(range(n_a))
        b_idx = list(range(n_b))
        rng.shuffle(a_idx)
        rng.shuffle(b_idx)
        a_sel = [a_rows[i] for i in a_idx[:take_a]]
        b_sel = [b_rows[i] for i in b_idx[:take_b]]

    merged = a_sel + b_sel
    rng.shuffle(merged)
    _write_json_array(out_path, merged)
    stats = {
        "source_a": str(source_a),
        "source_b": str(source_b),
        "source_a_rows": n_a,
        "source_b_rows": n_b,
        "selected_a": len(a_sel),
        "selected_b": len(b_sel),
        "selected_total": len(merged),
        "actual_ratio_a": float(len(a_sel) / max(1, len(merged))),
        "seed": int(seed),
        "sample_with_replacement": bool(sample_with_replacement),
        "out_path": str(out_path),
    }
    return out_path, stats


def _iter_all_dict_keys(obj: Any) -> Set[str]:
    keys: Set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys |= _iter_all_dict_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _iter_all_dict_keys(item)
    return keys


def validate_sanitized_hand(
    sanitized: Dict[str, Any],
    *,
    chunk_idx: int,
    hand_idx: int,
    collect_warnings: Optional[List[str]] = None,
) -> None:
    """Sanitized payload must not contain miner-forbidden keys; action grid matches miner contract."""
    leaked = _iter_all_dict_keys(sanitized) & MINER_LEAKAGE_KEYS
    if leaked:
        raise DataValidationError(
            f"Train/serve parity: leakage keys {sorted(leaked)} found after sanitize_hand_for_miner "
            f"(chunk={chunk_idx}, hand={hand_idx}). Labels must never appear in miner-visible JSON."
        )
    actions = sanitized.get("actions") or []
    if actions and len(actions) != EXPECTED_ACTION_SLOTS and collect_warnings is not None:
        collect_warnings.append(
            f"chunk {chunk_idx} hand {hand_idx}: expected 0 or {EXPECTED_ACTION_SLOTS} action slots, "
            f"got {len(actions)}"
        )


def validate_labeled_chunks(
    labeled_chunks: List[Dict[str, Any]],
    *,
    sample_hands_per_chunk: int = 2,
) -> Dict[str, Any]:
    """
    Checks raw labeled chunks before featurization:
      - every chunk has at least one hand
      - random sample of hands: sanitized view has no leakage keys
      - optional warnings for odd action-slot counts
    """
    warnings: List[str] = []
    if not labeled_chunks:
        raise DataValidationError("No labeled chunks produced.")

    empty = 0
    for ci, ch in enumerate(labeled_chunks):
        hands = ch.get("hands") or []
        if not hands:
            empty += 1
            continue
        if "is_bot" not in ch:
            warnings.append(f"chunk {ci}: missing 'is_bot' (treated as False)")
        for hi, h in enumerate(hands[: max(1, sample_hands_per_chunk)]):
            san = sanitize_hand_for_miner(h if isinstance(h, dict) else {})
            validate_sanitized_hand(san, chunk_idx=ci, hand_idx=hi, collect_warnings=warnings)

    if empty:
        raise DataValidationError(
            f"{empty}/{len(labeled_chunks)} chunks have zero hands — cannot build training rows."
        )

    return {
        "n_chunks": len(labeled_chunks),
        "chunks_with_hands": len(labeled_chunks) - empty,
        "warnings": warnings,
    }


def validate_feature_dataframe(df: pd.DataFrame, *, name: str = "dataframe") -> Dict[str, Any]:
    """
    Checks matrix ready for LightGBM / split:
      - non-empty, has label, binary {0,1}
      - both classes present (required for stratified split)
      - no NaN / inf in numeric feature columns
    """
    if df.empty:
        raise DataValidationError(f"{name}: empty DataFrame.")
    if "label" not in df.columns:
        raise DataValidationError(f"{name}: missing 'label' column.")
    labels = df["label"]
    if not labels.isin([0, 1]).all():
        bad = labels[~labels.isin([0, 1])].unique().tolist()
        raise DataValidationError(f"{name}: label must be 0 or 1; found {bad[:5]}.")

    n_bot = int(labels.sum())
    n_human = int(len(labels) - n_bot)
    if n_bot == 0 or n_human == 0:
        raise DataValidationError(
            f"{name}: need both human (0) and bot (1) rows for stratified split. "
            f"human={n_human}, bot={n_bot}."
        )

    feat = df.drop(columns=["label"])
    if feat.columns.duplicated().any():
        dup = feat.columns[feat.columns.duplicated()].tolist()
        raise DataValidationError(f"{name}: duplicate feature columns: {dup[:5]}.")

    na_cols = feat.columns[feat.isna().any()].tolist()
    if na_cols:
        raise DataValidationError(f"{name}: NaN in features (train/serve or bug): {na_cols[:12]}.")

    num = feat.select_dtypes(include=[np.number])
    if not num.empty and not np.isfinite(num.to_numpy(dtype=np.float64)).all():
        raise DataValidationError(f"{name}: non-finite values (inf/-inf) in numeric features.")

    return {"n_rows": len(df), "n_human": n_human, "n_bot": n_bot, "n_features": feat.shape[1]}


def validate_train_val_pair(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Same feature schema; both have valid labels and at least one row."""
    tc = [c for c in train_df.columns if c != "label"]
    vc = [c for c in val_df.columns if c != "label"]
    if tc != vc:
        raise DataValidationError(
            "Train/val feature column mismatch: "
            f"train_only={set(tc) - set(vc)!r} val_only={set(vc) - set(tc)!r}"
        )
    validate_feature_dataframe(train_df, name="train")
    validate_feature_dataframe(val_df, name="val")
    if len(val_df) == 0:
        raise DataValidationError("Validation split is empty.")


def align_dataframe_to_miner_features(
    df: pd.DataFrame,
    *,
    mode: MinerFeatureSchemaMode,
    name: str,
) -> pd.DataFrame:
    """
    Ensure columns match :func:`miner_servable_feature_names` plus ``label``.

    - ``strict``: exactly those feature columns (no extras — avoids miner zero-fill skew).
    - ``strip``: keep label + canonical features; drop extras with a log line.
    - ``off``: leave ``df`` unchanged.
    """
    if mode == "off" or df.empty:
        return df
    order = list(miner_servable_feature_names())
    expected = frozenset(order)
    if "label" not in df.columns:
        raise DataValidationError(f"{name}: missing 'label' column.")
    feat_names = [c for c in df.columns if c != "label"]
    actual = frozenset(feat_names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise DataValidationError(
            f"{name}: missing miner-servable feature columns "
            f"(not produced by aggregate_chunk_from_hands): {missing[:24]}"
            + (" ..." if len(missing) > 24 else "")
        )
    if mode == "strict" and extra:
        raise DataValidationError(
            f"{name}: extra columns not produced by aggregate_chunk_from_hands "
            f"(miner would fill them with 0.0 when aligning to the model): {extra[:24]}"
            + (" ..." if len(extra) > 24 else "")
        )
    if mode == "strip" and extra:
        print(
            f"[miner-schema] {name}: stripping {len(extra)} extra column(s): {extra[:12]!r}"
            + (" ..." if len(extra) > 12 else ""),
            flush=True,
        )
    return df[["label"] + order]


def load_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise DataValidationError(f"Unsupported file type: {path}")


def chunks_to_dataframe(
    labeled_chunks: List[Dict[str, Any]],
    *,
    progress_every: int = 100,
) -> pd.DataFrame:
    rows = []
    total = len(labeled_chunks)
    pe = max(1, int(progress_every))
    print(f"[featurize] start chunks={total}", flush=True)
    for i, ch in enumerate(labeled_chunks, start=1):
        hands = ch.get("hands") or []
        row = aggregate_chunk_from_hands(hands)
        row["label"] = int(bool(ch.get("is_bot")))
        rows.append(row)
        if i % pe == 0 or i == total:
            print(f"[featurize] {i}/{total} chunks", flush=True)
    return pd.DataFrame(rows)


def _combined_dataset_hash(shard_hashes: List[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(shard_hashes):
        h.update(s.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _write_partitioned_parquet(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    *,
    rows_per_file: int,
    format_ext: str,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    n = len(df)
    if rows_per_file <= 0 or n <= rows_per_file:
        p = out_dir / f"{prefix}.{format_ext}"
        if format_ext == "parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        paths.append(p)
        return paths
    part = 0
    for start in range(0, n, rows_per_file):
        chunk = df.iloc[start : start + rows_per_file]
        p = out_dir / f"{prefix}_{part:05d}.{format_ext}"
        if format_ext == "parquet":
            chunk.to_parquet(p, index=False)
        else:
            chunk.to_csv(p, index=False)
        paths.append(p)
        part += 1
    return paths


def merge_shards_and_split(
    *,
    out: Path,
    shards_dir: Path,
    val_size: float,
    seed: int,
    format: str,
    no_data_checks: bool,
    rows_per_partition: int,
    nested_train_val: bool,
    miner_feature_schema: MinerFeatureSchemaMode = "strict",
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Load shard_*.parquet from shards_dir, concat, stratified split, write train/val."""
    pattern = "shard_*.parquet" if format == "parquet" else "shard_*.csv"
    paths = sorted(shards_dir.glob(pattern))
    if not paths:
        raise DataValidationError(f"No {pattern} files under {shards_dir}")
    dfs = [load_frame(p) for p in paths]
    full = pd.concat(dfs, ignore_index=True)
    if "shard_idx" in full.columns:
        full = full.drop(columns=["shard_idx"])

    full = align_dataframe_to_miner_features(
        full, mode=miner_feature_schema, name="full_pre_split(sharded)"
    )

    if not no_data_checks:
        full_report = validate_feature_dataframe(full, name="full_pre_split")
        print(
            f"[data-check] rows={full_report['n_rows']} human={full_report['n_human']} "
            f"bot={full_report['n_bot']} features={full_report['n_features']}"
        )

    feature_cols = [c for c in full.columns if c != "label"]
    X = full[feature_cols]
    y = full["label"]
    try:
        print("[stage] train/val split ...", flush=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as e:
        raise DataValidationError(
            f"train_test_split failed ({e}). Often: too few samples for stratify, or single class."
        ) from e

    train_df = X_train.copy()
    train_df["label"] = y_train.values
    val_df = X_val.copy()
    val_df["label"] = y_val.values

    if not no_data_checks:
        validate_train_val_pair(train_df, val_df)
        print("[data-check] train/val split OK.")

    ext = "parquet" if format == "parquet" else "csv"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    shard_hashes = []
    meta_side = shards_dir / "shard_meta.json"
    if meta_side.exists():
        try:
            meta_js = json.loads(meta_side.read_text(encoding="utf-8"))
            shard_hashes = [s.get("dataset_hash", "") for s in meta_js.get("shards", [])]
        except (json.JSONDecodeError, OSError):
            shard_hashes = []
    ds_hash = _combined_dataset_hash([h for h in shard_hashes if h]) if shard_hashes else ""

    meta: Dict[str, Any] = {
        "mode": "sharded_merge",
        "dataset_hash": ds_hash,
        "shards_dir": str(shards_dir),
        "n_shard_files": len(paths),
        "n_rows": int(len(full)),
        "val_size": val_size,
        "seed": seed,
        "nested_train_val": nested_train_val,
        "rows_per_partition": rows_per_partition,
        "miner_feature_schema": miner_feature_schema,
        "miner_servable_feature_names": list(miner_servable_feature_names()),
    }

    if nested_train_val:
        train_dir = out / "train"
        val_dir = out / "val"
        rpp = max(0, int(rows_per_partition))
        train_written = _write_partitioned_parquet(
            train_df, train_dir, "train", rows_per_file=rpp, format_ext=ext
        )
        val_written = _write_partitioned_parquet(
            val_df, val_dir, "val", rows_per_file=rpp, format_ext=ext
        )
        meta["train_paths"] = [str(p) for p in train_written]
        meta["val_paths"] = [str(p) for p in val_written]
        train_path = train_written[0]
        val_path = val_written[0]
    else:
        train_path = out / f"train.{ext}"
        val_path = out / f"val.{ext}"
        print(f"[stage] writing {format} files ...", flush=True)
        if format == "parquet":
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
        else:
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
        meta["train_paths"] = [str(train_path)]
        meta["val_paths"] = [str(val_path)]

    meta["feature_cols"] = [c for c in train_df.columns if c != "label"]
    meta["n_train"] = int(len(train_df))
    meta["n_val"] = int(len(val_df))
    meta["sanitizer"] = "sanitize_hand_for_miner"

    (out / "manifest.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {train_path} ({len(train_df)} rows total train)")
    print(f"Wrote {val_path} ({len(val_df)} rows total val)")
    return train_path, val_path, meta


def _atomic_write_dataframe(
    df: pd.DataFrame,
    dest: Path,
    *,
    format: str,
) -> None:
    """Write parquet/csv to a temp file in the same directory, then replace (avoids half-written shards)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if format == "parquet" else ".csv"
    fd, tmp = tempfile.mkstemp(suffix=ext, prefix=".tmp_shard_", dir=str(dest.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        if format == "parquet":
            df.to_parquet(tmp_path, index=False)
        else:
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def build_sharded_training_dataset(
    *,
    out: Path,
    preset: PresetName = "training",
    chunk_count: Optional[int] = None,
    human_json: Optional[Path] = None,
    human_ratio: Optional[float] = None,
    min_hands: Optional[int] = None,
    max_hands: Optional[int] = None,
    seed: int = 42,
    val_size: float = 0.2,
    format: str = "parquet",
    no_data_checks: bool = False,
    bot_profile_mode: str = "mixed",
    bot_candidate_attempts: int = 8,
    bot_generation_rounds: int = 4,
    window_id: int = 0,
    progress_every_chunks: int = 100,
    shard_size: int = 1000,
    resume: bool = False,
    rows_per_partition: int = 0,
    nested_train_val: bool = False,
    miner_feature_schema: MinerFeatureSchemaMode = "strict",
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Build mixed data in shards under ``out/shards/``, then merge + one stratified split.
    Each shard uses an independent ``build_mixed_labeled_chunks`` call (decorrelated seed/window).
    """
    cc, mn, mx, hr = _effective_chunk_params(
        preset,
        chunk_count=chunk_count,
        min_hands=min_hands,
        max_hands=max_hands,
        human_ratio=human_ratio,
    )
    ss = max(1, int(shard_size))
    n_shards = (cc + ss - 1) // ss

    out = Path(out)
    _mkdir_output_dir(out)
    shards_dir = out / "shards"
    _mkdir_output_dir(shards_dir)

    shard_records: List[Dict[str, Any]] = []
    ext = "parquet" if format == "parquet" else "csv"

    for shard_idx in range(n_shards):
        start = shard_idx * ss
        sub_cc = min(ss, cc - start)
        shard_name = f"shard_{shard_idx:05d}.{ext}"
        shard_path = shards_dir / shard_name

        if resume and shard_path.exists():
            print(f"[shard] resume: skip existing {shard_path.name}", flush=True)
            if format == "parquet":
                tmp_df = pd.read_parquet(shard_path)
            else:
                tmp_df = pd.read_csv(shard_path)
            row_n = len(tmp_df)
            if row_n != sub_cc:
                print(
                    f"[shard] warning: {shard_path.name} has {row_n} rows but expected {sub_cc} "
                    f"(delete file to regenerate)",
                    flush=True,
                )
            ph = ""
            meta_p = shards_dir / f"shard_{shard_idx:05d}.meta.json"
            if meta_p.exists():
                try:
                    ph = json.loads(meta_p.read_text(encoding="utf-8")).get("dataset_hash", "")
                except (json.JSONDecodeError, OSError):
                    ph = ""
            shard_records.append(
                {
                    "shard_idx": shard_idx,
                    "chunk_count": row_n,
                    "dataset_hash": ph,
                    "path": str(shard_path),
                    "skipped": True,
                }
            )
            continue

        sub_seed = seed + shard_idx * _SHARD_SEED_STRIDE
        sub_window = window_id + shard_idx
        cfg_kw: Dict[str, Any] = {
            "chunk_count": sub_cc,
            "min_hands_per_chunk": mn,
            "max_hands_per_chunk": mx,
            "human_ratio": hr,
            "seed": sub_seed,
            "bot_profile_mode": bot_profile_mode,
            "bot_candidate_attempts_per_chunk": max(1, int(bot_candidate_attempts)),
            "max_bot_generation_rounds": max(1, int(bot_generation_rounds)),
        }
        if human_json is not None:
            cfg_kw["human_json_path"] = human_json
        cfg = MixedDatasetConfig(**filter_mixed_dataset_config_kwargs(cfg_kw))

        print(
            f"[shard {shard_idx + 1}/{n_shards}] building mixed labeled chunks "
            f"(sub_chunk_count={sub_cc}, window_id={sub_window}) ...",
            flush=True,
        )
        labeled_chunks, ds_hash, stats = build_mixed_labeled_chunks(cfg, window_id=sub_window)
        print(
            "[mixed-dataset] "
            f"chunks={stats.get('chunk_count')} "
            f"human_hands_in_chunks={stats.get('human_hands')} "
            f"bot_hands_in_chunks={stats.get('bot_hands')} "
            f"total_hands={stats.get('total_hands')}",
            flush=True,
        )

        if not no_data_checks:
            chunk_report = validate_labeled_chunks(labeled_chunks)
            for w in chunk_report.get("warnings", []):
                print(f"[data-check warning] {w}")

        df = chunks_to_dataframe(labeled_chunks, progress_every=progress_every_chunks)
        df = align_dataframe_to_miner_features(
            df,
            mode=miner_feature_schema,
            name=f"shard_{shard_idx:05d}",
        )
        df["shard_idx"] = shard_idx
        del labeled_chunks

        _atomic_write_dataframe(df, shard_path, format=format)

        (shards_dir / f"shard_{shard_idx:05d}.meta.json").write_text(
            json.dumps(
                {
                    "shard_idx": shard_idx,
                    "dataset_hash": ds_hash,
                    "stats": stats,
                    "sub_seed": sub_seed,
                    "window_id": sub_window,
                    "sub_chunk_count": sub_cc,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        shard_records.append(
            {
                "shard_idx": shard_idx,
                "chunk_count": sub_cc,
                "dataset_hash": ds_hash,
                "path": str(shard_path),
                "stats": stats,
                "skipped": False,
            }
        )

    (shards_dir / "shard_meta.json").write_text(
        json.dumps(
            {
                "total_chunk_count_requested": cc,
                "shard_size": ss,
                "n_shards": n_shards,
                "base_seed": seed,
                "base_window_id": window_id,
                "shards": shard_records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    combined_hash = _combined_dataset_hash([s["dataset_hash"] for s in shard_records if s.get("dataset_hash")])
    print(f"[stage] merged dataset_hash (from shards) = {combined_hash[:16]}...", flush=True)

    train_path, val_path, meta = merge_shards_and_split(
        out=out,
        shards_dir=shards_dir,
        val_size=val_size,
        seed=seed,
        format=format,
        no_data_checks=no_data_checks,
        rows_per_partition=rows_per_partition,
        nested_train_val=nested_train_val,
        miner_feature_schema=miner_feature_schema,
    )
    meta["mode"] = "sharded"
    meta["preset"] = preset
    meta["mixed_dataset_config"] = {
        "chunk_count": cc,
        "shard_size": ss,
        "min_hands_per_chunk": mn,
        "max_hands_per_chunk": mx,
        "human_ratio": hr,
        "bot_profile_mode": bot_profile_mode,
        "bot_candidate_attempts_per_chunk": max(1, int(bot_candidate_attempts)),
        "max_bot_generation_rounds": max(1, int(bot_generation_rounds)),
        "human_json_path": str(human_json or MixedDatasetConfig().human_json_path),
    }
    meta["dataset_hash"] = combined_hash
    meta["shard_meta"] = shard_records
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"manifest.json dataset_hash={combined_hash[:16]}...")
    return train_path, val_path, meta


def build_training_dataset(
    *,
    out: Path,
    preset: PresetName = "training",
    chunk_count: Optional[int] = None,
    human_json: Optional[Path] = None,
    human_ratio: Optional[float] = None,
    min_hands: Optional[int] = None,
    max_hands: Optional[int] = None,
    seed: int = 42,
    val_size: float = 0.2,
    format: str = "parquet",
    no_data_checks: bool = False,
    bot_profile_mode: str = "mixed",
    bot_candidate_attempts: int = 8,
    bot_generation_rounds: int = 4,
    window_id: int = 0,
    progress_every_chunks: int = 100,
    nested_train_val: bool = False,
    rows_per_partition: int = 0,
    miner_feature_schema: MinerFeatureSchemaMode = "strict",
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Build mixed labeled chunks → sanitized chunk features → stratified train/val on disk.

    Per-hand style: always ``sanitize_hand_for_miner``. Chunk counts/ratios follow ``preset``
    unless overridden (non-None keyword args win over preset).
    """
    cc, mn, mx, hr = _effective_chunk_params(
        preset,
        chunk_count=chunk_count,
        min_hands=min_hands,
        max_hands=max_hands,
        human_ratio=human_ratio,
    )

    cfg_kw: Dict[str, Any] = {
        "chunk_count": cc,
        "min_hands_per_chunk": mn,
        "max_hands_per_chunk": mx,
        "human_ratio": hr,
        "seed": seed,
        "bot_profile_mode": bot_profile_mode,
        "bot_candidate_attempts_per_chunk": max(1, int(bot_candidate_attempts)),
        "max_bot_generation_rounds": max(1, int(bot_generation_rounds)),
    }
    if human_json is not None:
        cfg_kw["human_json_path"] = human_json
    cfg = MixedDatasetConfig(**filter_mixed_dataset_config_kwargs(cfg_kw))

    print("[stage] building mixed labeled chunks ...", flush=True)
    labeled_chunks, ds_hash, stats = build_mixed_labeled_chunks(cfg, window_id=window_id)
    print(
        "[mixed-dataset] "
        f"chunks={stats.get('chunk_count')} "
        f"human_hands_in_chunks={stats.get('human_hands')} "
        f"bot_hands_in_chunks={stats.get('bot_hands')} "
        f"total_hands={stats.get('total_hands')}",
        flush=True,
    )

    check_report: Dict[str, Any] = {}
    if not no_data_checks:
        chunk_report = validate_labeled_chunks(labeled_chunks)
        check_report["labeled_chunks"] = chunk_report
        for w in chunk_report.get("warnings", []):
            print(f"[data-check warning] {w}")

    df = chunks_to_dataframe(labeled_chunks, progress_every=progress_every_chunks)
    df = align_dataframe_to_miner_features(
        df, mode=miner_feature_schema, name="full_pre_split"
    )

    if not no_data_checks:
        full_report = validate_feature_dataframe(df, name="full_pre_split")
        check_report["pre_split"] = full_report
        print(
            f"[data-check] rows={full_report['n_rows']} human={full_report['n_human']} "
            f"bot={full_report['n_bot']} features={full_report['n_features']}"
        )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    try:
        print("[stage] train/val split ...", flush=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as e:
        raise DataValidationError(
            f"train_test_split failed ({e}). Often: too few samples for stratify, or single class. "
            "Increase chunk_count or adjust human_ratio / val_size."
        ) from e

    train_df = X_train.copy()
    train_df["label"] = y_train.values
    val_df = X_val.copy()
    val_df["label"] = y_val.values

    if not no_data_checks:
        validate_train_val_pair(train_df, val_df)
        print("[data-check] train/val split OK.")

    out = Path(out)
    _mkdir_output_dir(out)
    meta: Dict[str, Any] = {
        "dataset_hash": ds_hash,
        "stats": stats,
        "preset": preset,
        "miner_feature_schema": miner_feature_schema,
        "miner_servable_feature_names": list(miner_servable_feature_names()),
        "mixed_dataset_config": {
            "chunk_count": cc,
            "min_hands_per_chunk": mn,
            "max_hands_per_chunk": mx,
            "human_ratio": hr,
            "bot_profile_mode": bot_profile_mode,
            "bot_candidate_attempts_per_chunk": max(1, int(bot_candidate_attempts)),
            "max_bot_generation_rounds": max(1, int(bot_generation_rounds)),
            "human_json_path": str(cfg.human_json_path),
        },
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "sanitizer": "sanitize_hand_for_miner",
        "data_checks": None if no_data_checks else check_report,
        "data_checks_skipped": bool(no_data_checks),
    }
    ext = "parquet" if format == "parquet" else "csv"
    rpp = max(0, int(rows_per_partition))
    if nested_train_val:
        train_dir = out / "train"
        val_dir = out / "val"
        print(f"[stage] writing {format} files under train/ and val/ ...", flush=True)
        train_written = _write_partitioned_parquet(
            train_df, train_dir, "train", rows_per_file=rpp, format_ext=ext
        )
        val_written = _write_partitioned_parquet(
            val_df, val_dir, "val", rows_per_file=rpp, format_ext=ext
        )
        train_path = train_written[0]
        val_path = val_written[0]
        meta["train_paths"] = [str(p) for p in train_written]
        meta["val_paths"] = [str(p) for p in val_written]
        meta["nested_train_val"] = True
        meta["rows_per_partition"] = rpp
    else:
        train_path = out / f"train.{ext}"
        val_path = out / f"val.{ext}"
        print(f"[stage] writing {format} files ...", flush=True)
        if format == "parquet":
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
        else:
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

    (out / "manifest.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {train_path} ({len(train_df)} rows)")
    print(f"Wrote {val_path} ({len(val_df)} rows)")
    print(f"Features ({len(feature_cols)}): {feature_cols[:6]!r} ...")
    print(f"manifest.json dataset_hash={ds_hash[:16]}...")

    return train_path, val_path, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Poker44 chunks for LightGBM (sanitized per-hand → chunk rows)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SUBNET_DEFAULT_OUT,
        help="Output directory for train.parquet / val.parquet (and shards/ when using --shard-size). "
        f"Default: {_rel_repo(SUBNET_DEFAULT_OUT)}",
    )
    parser.add_argument(
        "--preset",
        choices=cast(Tuple[str, ...], tuple(PRESETS.keys())),
        default="training",
        help="training=120 chunks; training-merged≈2920 chunks (use big merged human JSON); "
        "validator-parity=40. Each chunk → one Parquet row.",
    )
    parser.add_argument(
        "--chunk-count",
        type=int,
        default=None,
        help="Override preset: number of labeled chunks (= rows before train/val split). "
        "Larger = more human hands sampled from human_json (and more bot work).",
    )
    parser.add_argument(
        "--human-json",
        type=Path,
        default=SUBNET_DEFAULT_HUMAN_JSON,
        help="Human hands JSON or .gz. "
        f"Default: {_rel_repo(SUBNET_DEFAULT_HUMAN_JSON)} (--human-json-a/-b overrides via merged pool).",
    )
    parser.add_argument(
        "--human-json-a",
        type=Path,
        default=None,
        help="Source A human JSON/.gz for source-aware balancing (used with --human-json-b).",
    )
    parser.add_argument(
        "--human-json-b",
        type=Path,
        default=None,
        help="Source B human JSON/.gz for source-aware balancing (used with --human-json-a).",
    )
    parser.add_argument(
        "--human-source-ratio-a",
        type=float,
        default=0.5,
        help="Target fraction from source A when using --human-json-a/--human-json-b (default 0.5).",
    )
    parser.add_argument(
        "--human-balance-with-replacement",
        action="store_true",
        help="When balancing two human sources, sample with replacement to avoid truncation.",
    )
    parser.add_argument(
        "--human-ratio",
        type=float,
        default=None,
        help="Override preset: fraction of human chunks",
    )
    parser.add_argument("--min-hands", type=int, default=None, help="Override preset: min hands per chunk")
    parser.add_argument("--max-hands", type=int, default=None, help="Override preset: max hands per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split + dataset window")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format",
    )
    parser.add_argument(
        "--no-data-checks",
        action="store_true",
        help="Skip pre-training validation (not recommended)",
    )
    parser.add_argument(
        "--validate-files",
        nargs=2,
        metavar=("TRAIN", "VAL"),
        help="Only run data checks on existing train/val files (.parquet or .csv), then exit",
    )
    parser.add_argument(
        "--bot-profile-mode",
        choices=tuple(all_bot_profile_modes()),
        default="mixed",
        help="Bot behavior pool(s) for mixed dataset bot chunks",
    )
    parser.add_argument(
        "--bot-candidate-attempts",
        type=int,
        default=8,
        help="Bot generation candidates per chunk (lower = faster/less matching)",
    )
    parser.add_argument(
        "--bot-generation-rounds",
        type=int,
        default=4,
        help="Bot generation rounds to minimize shortcut leakage (lower = faster)",
    )
    parser.add_argument(
        "--window-id",
        type=int,
        default=0,
        help="Mixed dataset window_id (deterministic human sampling); default 0.",
    )
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=100,
        help="Print featurization progress every N chunks (default: 100).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, build in shards of N chunks under out/shards/, then merge + one stratified split. "
        "Use for large --chunk-count to reduce peak RAM and allow --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --shard-size: skip shard_XXXXX files that already exist.",
    )
    parser.add_argument(
        "--merge-only",
        type=Path,
        default=None,
        metavar="OUT_DIR",
        help="Do not generate data: load out/shards/shard_*.parquet from this output dir, "
        "concatenate, stratified split, write train/val (same as end of sharded run).",
    )
    parser.add_argument(
        "--nested-train-val",
        action="store_true",
        help="Write outputs under out/train/ and out/val/ (instead of out/train.parquet). "
        "Works for normal runs and for --shard-size / --merge-only.",
    )
    parser.add_argument(
        "--miner-feature-schema",
        choices=("strict", "strip", "off"),
        default="strict",
        help="Enforce chunk feature columns match aggregate_chunk_from_hands (miner inference). "
        "strict=no extra columns; strip=drop extras; off=legacy behavior.",
    )
    parser.add_argument(
        "--rows-per-partition",
        type=int,
        default=0,
        metavar="N",
        help="With --nested-train-val: split each of train/val into multiple part files with at most N rows "
        "(0 = one file per split).",
    )
    args = parser.parse_args()

    schema_mode = cast(MinerFeatureSchemaMode, str(args.miner_feature_schema))

    if args.validate_files:
        train_path, val_path = Path(args.validate_files[0]), Path(args.validate_files[1])
        train_df = load_frame(train_path)
        val_df = load_frame(val_path)
        train_df = align_dataframe_to_miner_features(
            train_df, mode=schema_mode, name="train(validate-files)"
        )
        val_df = align_dataframe_to_miner_features(
            val_df, mode=schema_mode, name="val(validate-files)"
        )
        validate_train_val_pair(train_df, val_df)
        print(f"✓ Data checks passed for {train_path.name} ({len(train_df)} rows) and {val_path.name} ({len(val_df)} rows).")
        return

    if args.merge_only is not None:
        out_dir = Path(args.merge_only).expanduser().resolve()
        shards_d = out_dir / "shards"
        if not shards_d.is_dir():
            raise DataValidationError(f"Missing or not a directory: {shards_d}")
        merge_shards_and_split(
            out=out_dir,
            shards_dir=shards_d,
            val_size=float(args.val_size),
            seed=int(args.seed),
            format=str(args.format),
            no_data_checks=bool(args.no_data_checks),
            rows_per_partition=max(0, int(args.rows_per_partition)),
            nested_train_val=bool(args.nested_train_val),
            miner_feature_schema=schema_mode,
        )
        return

    use_dual_sources = args.human_json_a is not None or args.human_json_b is not None
    human_json_resolved: Optional[Path] = None
    if use_dual_sources:
        if args.human_json_a is None or args.human_json_b is None:
            raise DataValidationError("Use both --human-json-a and --human-json-b together.")
        a = Path(args.human_json_a).expanduser().resolve()
        b = Path(args.human_json_b).expanduser().resolve()
        if args.human_json is not None:
            print(
                "[human-balance] --human-json ignored because --human-json-a/--human-json-b are set.",
                flush=True,
            )
        bal_seed = int(args.seed)
        bal_out = Path(args.out).expanduser().resolve() / "_balanced_human_pool.json"
        human_json_resolved, bal_stats = _build_balanced_human_pool(
            source_a=a,
            source_b=b,
            ratio_a=float(args.human_source_ratio_a),
            seed=bal_seed,
            out_path=bal_out,
            sample_with_replacement=bool(args.human_balance_with_replacement),
        )
        print(
            "[human-balance] "
            f"a={bal_stats['selected_a']} b={bal_stats['selected_b']} "
            f"total={bal_stats['selected_total']} ratio_a={bal_stats['actual_ratio_a']:.4f} "
            f"out={human_json_resolved}",
            flush=True,
        )
        bal_stats_path = Path(args.out).expanduser().resolve() / "human_balance_stats.json"
        bal_stats_path.parent.mkdir(parents=True, exist_ok=True)
        bal_stats_path.write_text(json.dumps(bal_stats, indent=2), encoding="utf-8")
        print(f"[human-balance] wrote stats: {bal_stats_path}", flush=True)
    elif args.human_json is not None:
        human_json_resolved = Path(args.human_json).expanduser().resolve()

    if args.shard_size is not None:
        build_sharded_training_dataset(
            out=args.out,
            preset=cast(PresetName, args.preset),
            chunk_count=args.chunk_count,
            human_json=human_json_resolved,
            human_ratio=args.human_ratio,
            min_hands=args.min_hands,
            max_hands=args.max_hands,
            seed=args.seed,
            val_size=args.val_size,
            format=args.format,
            no_data_checks=args.no_data_checks,
            bot_profile_mode=args.bot_profile_mode,
            bot_candidate_attempts=args.bot_candidate_attempts,
            bot_generation_rounds=args.bot_generation_rounds,
            window_id=max(0, int(args.window_id)),
            progress_every_chunks=max(1, int(args.progress_every_chunks)),
            shard_size=max(1, int(args.shard_size)),
            resume=bool(args.resume),
            rows_per_partition=max(0, int(args.rows_per_partition)),
            nested_train_val=bool(args.nested_train_val),
            miner_feature_schema=schema_mode,
        )
        return

    build_training_dataset(
        out=args.out,
        preset=cast(PresetName, args.preset),
        chunk_count=args.chunk_count,
        human_json=human_json_resolved,
        human_ratio=args.human_ratio,
        min_hands=args.min_hands,
        max_hands=args.max_hands,
        seed=args.seed,
        val_size=args.val_size,
        format=args.format,
        no_data_checks=args.no_data_checks,
        bot_profile_mode=args.bot_profile_mode,
        bot_candidate_attempts=args.bot_candidate_attempts,
        bot_generation_rounds=args.bot_generation_rounds,
        window_id=max(0, int(args.window_id)),
        progress_every_chunks=max(1, int(args.progress_every_chunks)),
        nested_train_val=bool(args.nested_train_val),
        rows_per_partition=max(0, int(args.rows_per_partition)),
        miner_feature_schema=schema_mode,
    )


if __name__ == "__main__":
    try:
        main()
    except DataValidationError as err:
        print(f"Data validation failed: {err}", file=sys.stderr)
        sys.exit(1)
