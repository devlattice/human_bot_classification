# SSL experiment hub (`workspace/ssl_data/SSL`)

Weak cluster pseudo-labels → `train.parquet` / `val.parquet` → `lgbm_2` (ultralow) → calibration → cross-eval.

Details and tuning notes:  
[`../usl_hdbscan/human_bot_validator/readme.md`](../usl_hdbscan/human_bot_validator/readme.md).

You need **`mixed_train_with_clusters.parquet`** before **§1** (or run **§0** first).

**Canonical script:** `../usl_hdbscan/human_bot_validator/prepare_weak_ssl_dataset.py`  
(shim: [`scripts/prepare_weak_ssl_dataset.py`](scripts/prepare_weak_ssl_dataset.py)).

---

## 0) From scratch: mix → HDBSCAN → join

Skip this block if **`workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet`** is already the mix you want.

```bash
cd /home/dr/Workspace/Poker44-subnet   # adjust if needed

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust/train.parquet \
  --validator-source workspace/ssl_data/raw_data/feature_1/requests_robusted.parquet \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \
  --output-name mixed_train.parquet \
  --n-per-class auto \
  --seed 42 \
  --summary \
  --copy-manifest

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --min-cluster-size 80 \
  --min-samples 40 \
  --metric euclidean \
  --cluster-selection-epsilon 0.0 \
  --random-state 42

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/join_mixed_clusters.py \
  --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet
```

---

## Customize (edit once, reuse)

Run from **repo root**. Set variables to match your machine:

```bash
cd /home/dr/Workspace/Poker44-subnet   # or: export REPO=/path/to/Poker44-subnet && cd "$REPO"

# --- inputs (change if your layout differs) ---
export MIXED_WIDE=workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet
export VAL_PARQUET=workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust/val.parquet

# --- this run’s tag (folder names) ---
export SSL_TAG=ssl_weak_step1

# --- weak SSL knobs ---
export PSEUDO_WEIGHT=0.15      # try 0.05, 0.15, 0.25
export MIN_CLUSTER_PROB=0.7    # try 0.55 (looser), 0.7 (baseline), 0.8, 0.9
export SEED=42

# --- outputs ---
export SSL_DATA_DIR=workspace/ssl_data/SSL/data/${SSL_TAG}
export ARTIFACT_DIR=workspace/model/artifacts/lgbm_weak_ssl_ultralow_${SSL_TAG}
export CROSS_EVAL_DIR=workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature56_ssl_ultralow_${SSL_TAG}_raw
```

**Shipped baseline in evals:** `SSL_TAG=ssl_weak_step1`, `PSEUDO_WEIGHT=0.15`, `MIN_CLUSTER_PROB=0.7`.  
**Looser cluster gate (example):** `SSL_TAG=ssl_weak_mcp055`, `MIN_CLUSTER_PROB=0.55` (new dirs; do not overwrite `ssl_weak_step1`).

## 1) Prepare weak SSL dataset

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/prepare_weak_ssl_dataset.py \
  --input "${MIXED_WIDE}" \
  --out-dir "${SSL_DATA_DIR}" \
  --val-parquet "${VAL_PARQUET}" \
  --pseudo-weight "${PSEUDO_WEIGHT}" \
  --min-cluster-prob "${MIN_CLUSTER_PROB}" \
  --seed "${SEED}"
```

**Ablations (examples):**

- **No pseudo:** add `--no-pseudo` (ignore `PSEUDO_WEIGHT` / `MIN_CLUSTER_PROB` for labeling).
- **Stricter pseudo:** raise `MIN_CLUSTER_PROB` (fewer pseudo rows).
- **Looser pseudo:** lower `MIN_CLUSTER_PROB` (e.g. `0.55` — more rows, noisier labels; use a new `SSL_TAG`).
- **Weaker pseudo in loss:** lower `PSEUDO_WEIGHT`.
- **Optional:** `--agreement logistic`, `--pseudo-fraction 0.25`.

Check counts: `"${SSL_DATA_DIR}/ssl_prepare_summary.json"`.

---

## 2) Train `lgbm_2` (ultralow, matches `lgbm_feature56_v4_ultralow`)

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \
  --data-dir "${SSL_DATA_DIR}" \
  --sample-weight-col sample_weight \
  --out-dir "${ARTIFACT_DIR}" \
  --n-estimators 8000 \
  --learning-rate 0.01 \
  --num-leaves 3 \
  --max-depth 2 \
  --min-child-samples 2000 \
  --subsample 0.4 \
  --colsample-bytree 0.4 \
  --reg-alpha 10.0 \
  --reg-lambda 30.0 \
  --min-gain-to-split 0.35 \
  --early-stopping-rounds 250
```

To change capacity, edit the `lgbm_2` flags; keep **`--data-dir`** aligned with **`SSL_DATA_DIR`**.

---

## 3) Calibrate (Platt on training val)

Uses **`${SSL_DATA_DIR}/val.parquet`** (same split as training).

```bash
PYTHONPATH=. python workspace/model/scripts/calibrate_lgbm_joblib.py \
  --artifact-dir "${ARTIFACT_DIR}" \
  --data-dir "${SSL_DATA_DIR}" \
  --method sigmoid \
  --report-threshold 0.5
```

Writes **`lgbm_b_classifier_calibrated.joblib`** and **`.calibration.json`** under **`${ARTIFACT_DIR}`**.

---

## 4) Cross-eval (raw vs calibrated)

**Raw** (`lgbm_b_classifier.joblib`):

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model "${ARTIFACT_DIR}/lgbm_b_classifier.joblib" \
  --metrics-json "${ARTIFACT_DIR}/metrics.json" \
  --threshold 0.5 \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir "${CROSS_EVAL_DIR}"
```

**Calibrated** — set e.g. `export CROSS_EVAL_DIR=.../feature56_ssl_ultralow_${SSL_TAG}_calibrated` and:

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model "${ARTIFACT_DIR}/lgbm_b_classifier_calibrated.joblib" \
  --metrics-json "${ARTIFACT_DIR}/metrics.json" \
  --threshold 0.5 \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir "${CROSS_EVAL_DIR}"
```

To score **other** parquets, replace or extend **`--eval-parquet`** lines. Compare **`cross_dataset_comparison.csv`** across runs.

---

## 5) Knob sweep (1–2 runs)

Use a **new `SSL_TAG`** per run so data and artifacts do not overwrite.

| Run | `SSL_TAG` suggestion | `PSEUDO_WEIGHT` | `MIN_CLUSTER_PROB` |
|-----|----------------------|-----------------|----------------------|
| Baseline | `ssl_weak_step1` | `0.15` | `0.7` |
| Looser cluster gate | `ssl_weak_mcp055` | `0.15` | `0.55` |
| Softer pseudo | `ssl_weak_step1_pw005` | `0.05` | `0.7` |
| Stricter cluster | `ssl_weak_step1_mcp08` | `0.15` | `0.8` |
| Both | `ssl_weak_step1_pw005_mcp08` | `0.05` | `0.8` |

For each row: set `SSL_TAG`, `PSEUDO_WEIGHT`, `MIN_CLUSTER_PROB` in **Customize**, then §1 → §2 → §4 (raw cross-eval). Calibrate (§3) when you want **calibrated** metrics.

Stop when **holdout** `val_human_fpr` / `val_roc_auc` / `val_log_loss` at **0.5** stop improving vs your baseline CSV.

---

## 6) Miner

Point **`POKER44_MINER_MODEL_PATH`** at **`${ARTIFACT_DIR}/lgbm_b_classifier_calibrated.joblib`** (or raw joblib). Miner uses **`>= 0.5`** in `neurons/miner.py`.

---

## Layout

| Path | Contents |
|------|----------|
| `SSL/data/<tag>/` | `train.parquet`, `val.parquet`, `ssl_prepare_summary.json` |
| `workspace/model/artifacts/lgbm_weak_ssl_ultralow_<tag>/` | `metrics.json`, joblibs, plots |
| `.../cross_eval/feature56_ssl_ultralow_<tag>_raw/` (or `_calibrated`) | `cross_dataset_comparison.csv` |
