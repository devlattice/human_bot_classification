# `usl_hdbscan` — unsupervised labeling prep (JSONL → robust Parquet)

Build **chunk-level** tables from miner **JSONL** logs so you can run **HDBSCAN** (or other USL) and plot **pseudo-labels** (human / bot / noise) against **`miner_score`** (raw `risk_score` from the log line).

This folder is separate from generic SSL tooling in `workspace/ssl_data/readme.md`: here the focus is **validator traffic features** + **miner score** for exploration, not training labels from `risk_score` buckets.

---

## What gets built

| Output | Contents |
|--------|----------|
| **`usl.parquet`** (default `--output-name`) | `label` (all null) + **robust features** (same order as `--same-schema`) + **`miner_score`** last. Use for **plots** (e.g. cluster vs miner score). |
| **`usl_hdbscan_features.parquet`** (default `--hdbscan-parquet-name`) | **Only** the robust feature columns — **no** `label`, **no** `miner_score`. Use this file **as the matrix for HDBSCAN** so miner score cannot leak into clustering. |
| **`build_manifest.json`** | Row counts, `hdbscan_feature_columns`, and `hdbscan_safe` paths. |

**Pipeline:** `aggregate_chunk_from_hands` → drop list → **`transform_meta.json` apply-only** (clip / log1p / robust scale / fillna).  
**`miner_score` is never passed into the robust transform** (it is not in `clip_bounds`).

---

## Requirements

- **Python:** `pandas`, **`pyarrow`** (batched Parquet writes).
- **`--same-schema`:** a **robusted** `train.parquet` whose columns are `label` + your keep features (defines feature **order** and count).
- **`--transform-meta`:** the **`transform_meta.json`** that matches that training pipeline.
- **Clustering:** `pip install hdbscan scikit-learn` (for `cluster_hdbscan.py`).

**Naming:** do not create a file named `hdbscan.py` in this folder — it shadows the PyPI **`hdbscan`** package and breaks clustering. Use **`cluster_hdbscan.py`** only.

---

## Commands

Run from the **repo root** (`Poker44-subnet`) with `PYTHONPATH=.` so `poker44` imports resolve.

### Default paths (json dir + system_human_bot schema + output under `data/`)

```bash
cd /path/to/Poker44-subnet

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/build_usl_data.py
```

This reads all `workspace/ssl_data/json/*.jsonl`, uses:

- `--same-schema workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet`
- `--transform-meta workspace/dataset/robusted_dataset/train/system_human_bot/transform_meta.json`
- `--output-dir workspace/ssl_data/usl_hdbscan/data`

### Explicit JSONL files and output directory

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/build_usl_data.py \
  --json-files workspace/ssl_data/json/logs_a.jsonl workspace/ssl_data/json/logs_b.jsonl \
  --same-schema workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet \
  --transform-meta workspace/dataset/robusted_dataset/train/system_human_bot/transform_meta.json \
  --output-dir workspace/ssl_data/usl_hdbscan/data \
  --output-name usl.parquet \
  --batch-size 2048 \
  --hdbscan-parquet-name usl_hdbscan_features.parquet
```
```bash
--json-dir workspace/ssl_data/json \
```
### CPU / memory (batched streaming)

Peak RAM scales with **`--batch-size`** (default `2048`), not total JSONL size. Lower on small machines, raise if you have headroom:

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/build_usl_data.py --batch-size 1024
```

### Optional: `keep_features.txt` / `drop_features.txt`

If you pass **`--feature-selection /path/to/dir`**, that directory must contain **`keep_features.txt`** and **`drop_features.txt`**, and **`keep_features.txt` must match** the feature column names and order implied by `--same-schema`** (excluding `label`).

### Disable the HDBSCAN-only file

If you only want the full parquet (not recommended for clustering safety):

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/build_usl_data.py --hdbscan-parquet-name ""
```

### Help

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/build_usl_data.py --help
```

---

## HDBSCAN input (important)

- Load **`usl_hdbscan_features.parquet`** (or columns listed in `build_manifest.json` → `hdbscan_feature_columns`).
- Optionally **standardize** or reduce dimensionality **on that matrix only**; do **not** append `miner_score` unless you intentionally want it to influence geometry.
- Use **`usl.parquet`** when you need **`miner_score`** aligned row-by-row with the same chunk order for visualization.

### Run clustering (`cluster_hdbscan.py`)

Writes **`clusters.parquet`** (`cluster`, optional `cluster_probability`) and **`clusters.json`** metadata — **same row order** as the feature input (join with `plot.py` via `--clusters-parquet`).

```bash
pip install hdbscan scikit-learn

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/data/usl_hdbscan_features.parquet \
  --output workspace/ssl_data/usl_hdbscan/data/clusters.parquet \
  --min-cluster-size 15
```

If `build_manifest.json` sits in the **same directory** as `--input`, feature names are taken from `hdbscan_feature_columns` automatically (same as passing `--manifest` explicitly).

- **`--manifest`:** override path to `build_manifest.json` when it is not beside the input.
- Omit **`--manifest`** (and no sidecar file) and **`--columns`** → auto-select all numeric columns (excluding `label` / `miner_score` / `cluster`).
- **`--no-scale`:** skip `StandardScaler` if you already want raw robust features in distance space.

### Pseudo human / bot from clusters + miner score

After you inspect `miner_score_by_cluster.png`, map each **cluster id** to **pseudo_human** / **pseudo_bot** / **uncertain** using **per-cluster median** `miner_score` (noise `-1` is always **uncertain**):

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/assign_pseudo_labels.py \
  --usl-parquet workspace/ssl_data/usl_hdbscan/data/usl.parquet \
  --clusters-parquet workspace/ssl_data/usl_hdbscan/data/clusters.parquet \
  --bot-median-min 0.65 \
  --human-median-max 0.35
```

Outputs `data/pseudo_label_map.parquet` (cluster → label + medians) and `data/usl_with_pseudo.parquet` (all `usl` columns + `pseudo_label`). **Tune** `--bot-median-min` and `--human-median-max` using your violin plot so high-score clusters become bot and low-score become human; the band between the two thresholds stays **uncertain**.

---

## Layout

| Path | Role |
|------|------|
| `build_usl_data.py` | Builder CLI |
| `cluster_hdbscan.py` | HDBSCAN on feature Parquet → `clusters.parquet` |
| `assign_pseudo_labels.py` | cluster → pseudo_human / pseudo_bot / uncertain (from miner_score) |
| `plot.py` | `miner_score` vs cluster / label plots |
| `data/` | Default output (`usl.parquet`, `usl_hdbscan_features.parquet`, `build_manifest.json`) |

---

## Input JSONL shape

Same as other miner SSL tooling: one JSON object per line with at least **`chunk`** (hand list), **`chunk_hash`** (for dedup), and **`risk_score`** (skipped rows if missing or invalid).
