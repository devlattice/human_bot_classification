# SSL / semi-supervised data (`ssl_data`)

Tools for building extra training data from miner-logged chunks scored by the live model (`risk_score`).

## Layout

| Path | Role |
|------|------|
| `json/` | Drop **`*.jsonl`** (and optionally `*.json`) inputs: one JSON object per line. |
| `split_by_score.py` | Pipeline script: dedup + split by score → three JSON array files. |

## Input format

Each line should be a JSON object with at least:

- `chunk_hash` — stable id (used for deduplication)
- `chunk` — list of hand dicts (miner-visible schema)
- `risk_score` — float in `[0, 1]` (e.g. from `POKER44_MINER_LOG_CHUNK_NDJSON`)

Files are read in **sorted filename order**. Duplicate `chunk_hash` values after the first are skipped when dedup is on.

## `split_by_score.py`

Writes **`score_low.json`**, **`score_medium.json`**, **`score_high.json`** under `--output-json-dir`. Each file is a **JSON array** of the same record objects.

**Buckets** (requires `low_threshold < high_threshold`):

- **Low:** `risk_score <= low_threshold`
- **High:** `risk_score >= high_threshold`
- **Medium:** `low_threshold < risk_score < high_threshold`

### CLI

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-json-dir` | `ssl_data/json` (next to the script) | Directory of `*.jsonl` / `*.json` |
| `--output-json-dir` | *(required)* | Output directory for the three JSON files |
| `--low-score-threshold` | `0.33` | Upper bound for low bucket (inclusive) |
| `--high-score-threshold` | `0.67` | Lower bound for high bucket (inclusive) |
| `--no-dedup` | off | Disable deduplication by `chunk_hash` within input |
| `--replace-output` | off | Do **not** load existing `score_*.json`; output only from input |
| `--pretty` | off | Pretty-print JSON (slower / larger; default is compact) |
| `--quiet` | off | Less stderr logging |
| `--log-every` | `100000` | Progress log every N records per input file (`0` = off) |

### Merge vs replace

**Default:** Existing **`score_low.json`**, **`score_medium.json`**, **`score_high.json`** in `--output-json-dir` are loaded (must be JSON **arrays**). Records are merged by `chunk_hash`: **first seen** wins when loading existing files (order: low → medium → high). New input lines only add hashes **not** already present. Everything is then **re-bucketed** with the current thresholds. **No other files** in the output directory are deleted or touched.

**`--replace-output`:** Skip loading existing outputs; build maps only from input (still overwrites only the three `score_*.json` files).

### Example

From repo root:

```bash
python3 workspace/datasets/ssl_data/split_by_score.py \
  --input-json-dir workspace/datasets/ssl_data/json \
  --output-json-dir workspace/datasets/ssl_data/split_out \
  --low-score-threshold 0.2 \
  --high-score-threshold 0.8
```

Invalid lines, missing `chunk_hash`, or non-numeric `risk_score` are skipped with warnings on stderr.

### Note on `.json` files

Files with a `.json` extension are still read as **one JSON object per line** (JSONL-style), not as a single large JSON array.

## `build_dataset.py`

Builds chunk-level **train** / **val** tables aligned with the same **`keep_features` + `transform_meta`** stack you use for robust / miner inference (e.g. `workspace/model/artifacts/lgbm_keep` or `robusted_dataset/balanced_sources_35k_keep`):

1. Load `score_low.json` + `score_high.json` (JSON arrays).
2. One row per chunk: `aggregate_chunk_from_hands` (same as miner / `split_by_score` source).
3. **Label:** default **source-based** — low file → `0`, high file → `1`. Optional `--label-mode threshold --risk-threshold 0.5` binarizes `risk_score` instead.
4. Drop columns in `drop_features.txt`; **apply-only** `transform_meta.json` (clip / log1p / robust scale / fillna from `robust_feature_transform.py`).
5. Keep columns in `keep_features.txt` + `label`, in the same order as `robust_feature_transform --restrict-to-keep-features` (`label` first). **`chunk_hash` and teacher `risk_score` are never written** to train/val tables (training parity with e.g. `robusted_dataset/balanced_sources_35k_keep/train.parquet`; avoids leakage and serve-time mismatch).
6. Stratified `train_test_split`.

**Feature matching:** **`--strict-features`** is on by default — the run **fails** if any name in `keep_features.txt` is missing after `transform_meta`. Use **`--no-strict-features`** to only warn. **`--match-columns-parquet`** reads the reference **schema only** (no row decode; light on CPU/RAM). With strict mode, it is compared to `keep_features.txt` **before** loading the large score JSON files. Schema read needs **`pyarrow`** or **`fastparquet`** (`pip install pyarrow`). Writing **`train.parquet` / `val.parquet`** still usually needs **`pyarrow`** (or use **`--csv`**).

Optional **`--refine-score-json`**: path to a JSON array file (e.g. **`refine-score-json.json`**). Each object **must** include **`label`** (**0** or **1**) and **`chunk`**. Optional **`chunk_hash`**, **`risk_score`**. Same drop / `transform_meta` / keep pipeline as tail data. Use **raw miner-view chunks**, not pre-transformed feature vectors (avoid double application).

Requires **`PYTHONPATH=<repo root>`** (or run from repo with `python -m` after adjusting paths). Parquet needs **`pyarrow`**; use **`--csv`** if pyarrow is not installed.

**Important:** `--feature-selection` (directory with `keep_features.txt` + `drop_features.txt`) must belong to the **same** preprocessing run as **`--transform-meta`**. If you point at another tree’s `transform_meta.json`, use that run’s feature-selection directory (e.g. `workspace/_subnet_target/preprocess/feature_selection` when that’s the matching pair).

### Example (from repo root)

```bash
cd /path/to/miner_1

PYTHONPATH=. python3 workspace/datasets/ssl_data/build_dataset.py \
  --low-score-json workspace/datasets/ssl_data/split_out/score_low.json \
  --high-score-json workspace/datasets/ssl_data/split_out/score_high.json \
  --feature-selection workspace/test/feature_selection \
  --transform-meta workspace/datasets/robusted_dataset/balanced_sources_35k_keep/transform_meta.json \
  --output workspace/datasets/ssl_data/robusted_ssl_out \
  --val-fraction 0.2
```

If your artifacts live under **`workspace/model/artifacts/lgbm_keep`**, swap `--transform-meta` (and **`--feature-selection`**) to that bundle’s paths instead.

Outputs **`train.parquet`**, **`val.parquet`**, and **`build_manifest.json`** under `--output`.

### Optional flags

```bash
# Refined middle-band (or mixed) chunks: each record needs `label` (0/1) and `chunk` (raw hands).
  --refine-score-json workspace/datasets/ssl_data/refine-score-json.json

# Exact column names + order vs reference train.parquet (schema only; pyarrow or fastparquet).
  --match-columns-parquet workspace/datasets/robusted_dataset/balanced_sources_35k_keep/train.parquet

# Allow missing keep names with warnings only (default is strict: fail if any keep is missing).
  --no-strict-features

# If parquet engine is missing: write CSV instead.
  --csv
```

End-to-end after splitting scores:

```bash
python3 workspace/datasets/ssl_data/split_by_score.py \
  --input-json-dir workspace/datasets/ssl_data/json \
  --output-json-dir workspace/datasets/ssl_data/split_out \
  --low-score-threshold 0.2 \
  --high-score-threshold 0.8

PYTHONPATH=. python3 workspace/datasets/ssl_data/build_dataset.py \
  --low-score-json workspace/datasets/ssl_data/split_out/score_low.json \
  --high-score-json workspace/datasets/ssl_data/split_out/score_high.json \
  --feature-selection workspace/test/feature_selection \
  --transform-meta workspace/datasets/robusted_dataset/balanced_sources_35k_keep/transform_meta.json \
  --output workspace/datasets/ssl_data/robusted_ssl_out \
  --val-fraction 0.2
```
