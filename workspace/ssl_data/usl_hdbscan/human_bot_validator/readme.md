# Human / bot / validator mix → HDBSCAN → weak SSL

`cluster_hdbscan.py` applies **`StandardScaler`** then Euclidean HDBSCAN by default. Use **`--no-scale`** only if distances should use the input scale as-is.

---

## 1. `mix_data.py`

Samples **`n`** rows per class from labeled train, appends **all** validator rows, shuffles. Writes **`mixed_train.parquet`** + `mix_source` (`train_human` / `train_bot` / `validator`).

Default PNGs go to **`<--output-dir>/plots/`** (`--no-plot` to skip, `--plot-dir` to override).

**feature_2 example** (validator often has fewer columns than train — use **`--intersect-features`**):

```bash
cd /path/to/Poker44-subnet

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet \
  --validator-source workspace/ssl_data/raw_data/feature_2/validator.parquet \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/data \
  --output-name mixed_train.parquet \
  --n-per-class auto \
  --intersect-features \
  --seed 42 \
  --summary \
  --copy-manifest
```

- **`--n-per-class auto`:** use `min(count_human, count_bot)` from `--real-source`.
- **Schema mismatch:** without `--intersect-features`, validator must include every column from `--real-source`. With it, only the **intersection** is kept (train column order); stderr lists dropped train-only columns. If you **`--copy-manifest`**, trim `hdbscan_feature_columns` in the copied `build_manifest.json` to match intersected features, or rely on `cluster_hdbscan` numeric auto-selection.

### Optional extra sources (e.g. IRC)

`mix_data.py` can append up to two extra parquets:
- `--extra-source-1`, `--extra-source-1-rate`
- `--extra-source-2`, `--extra-source-2-rate`

Rate is relative to sampled labeled rows (`2 * n_per_class`):  
e.g. `--extra-source-1-rate 0.2` adds about `0.2 * (2n)` rows from extra source 1 (capped by available rows).

Extra rows are tagged as `mix_source=extra_source_1` / `extra_source_2`.

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet \
  --validator-source workspace/ssl_data/raw_data/feature_2/validator.parquet \
  --extra-source-1 workspace/preprocess/statistical_test/explorer/feature_2/data/irc/irc_train.parquet \
  --extra-source-1-rate 0.2 \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/data \
  --n-per-class auto \
  --intersect-features \
  --summary
```

---

## 2. `cluster_hdbscan.py`

Writes **`mixed_clusters.parquet`** (same row order as `--input`), sidecar **`mixed_clusters.json`**, and PNGs under **`<output parent>/plots/`** unless **`--no-plot`**.

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_clusters.parquet \
  --min-cluster-size 80 \
  --min-samples 40 \
  --random-state 42
```

Omit **`--min-samples`** to default it to **`--min-cluster-size`**. Uses **`build_manifest.json`** beside `--input` when present (`--manifest` to override). **`--no-scale`** skips the extra scaler.

---

### Grid search (train gate → minimize validator noise)

This mode enforces:
- `train_balanced_accuracy_clustered >= --min-train-balanced-accuracy`
- `train_clustered_coverage >= --min-train-clustered-coverage`

Among feasible points, it selects the one with lowest `validator_noise_frac` and writes:
- final `mixed_clusters.parquet`
- grid table CSV (`--grid-csv`)
- live progress logs in shell (`--grid-log-every`)

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_clusters.parquet \
  --grid-search \
  --grid-min-cluster-size-start 20 \
  --grid-min-cluster-size-stop 100 \
  --grid-min-cluster-size-step 10 \
  --grid-min-samples-start 5 \
  --grid-min-samples-stop 50 \
  --grid-min-samples-step 10 \
  --min-train-balanced-accuracy 0.80 \
  --min-train-clustered-coverage 0.70 \
  --grid-log-every 1 \
  --grid-csv workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_clusters.grid.csv \
  --random-state 42
```

---

### Confusion-style cluster plots
```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/plot_mixed_clusters.py \
  --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train.parquet \
  --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_clusters.parquet \
  --out-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/data/plots
```

## 3. `join_mixed_clusters.py`

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/join_mixed_clusters.py \
  --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train.parquet \
  --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_clusters.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train_with_clusters.parquet
```

---

## 4. `prepare_weak_ssl_dataset.py`

Builds **`train.parquet`** (real labels + optional validator pseudo-labels + `sample_weight`) and **`val.parquet`**. Requires **`mixed_train_with_clusters.parquet`**.

Typical call: external labeled val so **all** mixed human/bot rows stay in train:

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/prepare_weak_ssl_dataset.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/data/mixed_train_with_clusters.parquet \
  --out-dir workspace/ssl_data/SSL/data/ssl_47_mcp055 \
  --val-parquet workspace/preprocess/statistical_test/explorer/feature_2/data/public/val.parquet \
  --min-cluster-prob 0.55 \
  --pseudo-weight 0.15 \
  --seed 42
```

Adjust **`--val-parquet`** / **`--out-dir`** to your explorer layout. Without **`--val-parquet`**, use **`--val-fraction`** on labeled mixed rows only. Flags: **`--no-pseudo`**, **`--pseudo-fraction`**, **`--agreement logistic`**, **`--cluster-human` / `--cluster-bot`**.

**Train:** `workspace/model/scripts/lgbm_2.py` with **`--sample-weight-col sample_weight`** and the same **`--data-dir`** as `--out-dir` above. Optional calibration: `calibrate_lgbm_joblib.py` — see `workspace/model/readme.md`.

---

## 5. `plot_mixed_clusters.py`

Heatmap **`mix_source × cluster`** and label-fraction bars (needs matplotlib + seaborn). Default **`--out-dir`** is this folder; set it e.g. to **`.../plots`**.

---

## 6. Artifacts

| Path | Role |
|------|------|
| `mixed_train.parquet` | Features + `label` + `mix_source` |
| `mixed_clusters.parquet` | `cluster`, `cluster_probability` (aligned rows) |
| `mixed_train_with_clusters.parquet` | Wide join |
| `mixed_clusters.json` | HDBSCAN meta + `plot_paths` |
| `plots/` | `mix_data` + `cluster_hdbscan` PNGs |
| `ssl_weak_step1/` (or chosen `--out-dir`) | `train.parquet`, `val.parquet`, `ssl_prepare_summary.json` |
