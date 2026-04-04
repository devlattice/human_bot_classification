# Human / bot / validator mix → HDBSCAN

**Scaling:** `cluster_hdbscan.py` uses **`StandardScaler` (z-score per feature) on the full mixed matrix** before Euclidean HDBSCAN **by default**. To disable: pass **`--no-scale`**.

## 1) Mix labeled train + validator requests

Use **`--n-per-class auto`** if you want the maximum balanced sample from real train (e.g. 4000+4000 when that is the limit). Validator rows are appended in full.

Paths assume **57 features + `label`** (same columns in both parquets; validator may have `label` NA — ok).

```bash
cd /home/dr/Workspace/Poker44-subnet

PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/mix_data.py \
  --real-source workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust/train.parquet \
  --validator-source workspace/ssl_data/raw_data/feature_1/requests_robusted.parquet \
  --output-dir workspace/ssl_data/usl_hdbscan/human_bot_validator \
  --output-name mixed_train.parquet \
  --n-per-class auto \
  --seed 42 \
  --summary \
  --copy-manifest
```

Fixed sample size (e.g. 3000 per class) instead of `auto`:

```bash
  --n-per-class 3000
```

## 2) HDBSCAN with StandardScaler (default)

Uses **`build_manifest.json`** next to `--input` if present (from `--copy-manifest`), for `hdbscan_feature_columns`. Otherwise all numeric columns except `label`, `mix_source`, etc.

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/cluster_hdbscan.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --min-cluster-size 80 \
  --min-samples 40 \
  --metric euclidean \
  --cluster-selection-epsilon 0.0 \
  --random-state 42
```

**Tuning hints** (~14k rows mixed):

| Argument | Role |
|----------|------|
| `--min-cluster-size` | Larger → fewer, bigger clusters; try **50–150** first. |
| `--min-samples` | Defaults to `min_cluster_size` if omitted; **lower** (e.g. half of min_cluster_size) → denser cores, often **less noise**. |
| `--metric` | Keep **`euclidean`** with StandardScaler (default). |
| `--cluster-selection-epsilon` | **0** default; small **>0** merges close clusters (use sparingly). |

**Explicit manifest** (if not beside `mixed_train.parquet`):

```bash
  --manifest workspace/ssl_data/usl_hdbscan/data/build_manifest.json
```

**Skip StandardScaler** (only if you explicitly want raw mixed features in distance):

```bash
  --no-scale
```

## 3) Join train + cluster columns (wide parquet)

Row order is the same as `mixed_train.parquet` and `mixed_clusters.parquet`; this script concatenates horizontally.

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/join_mixed_clusters.py \
  --mixed workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train.parquet \
  --clusters workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_clusters.parquet \
  --output workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet
```

Defaults: all three paths under `human_bot_validator/` with those filenames.

## 4) Weak SSL dataset → `train.parquet` / `val.parquet` (step-by-step)

Do this **after** §3 so you have `mixed_train_with_clusters.parquet`.

| Step | What |
|------|------|
| 1 | Load wide mixed + clusters. |
| 2 | **Real labels:** `train_human` / `train_bot` rows with non-null `label`. |
| 3 | **Val:** either stratified holdout from labeled mixed (`--val-fraction`) **or** `--val-parquet` (e.g. explorer `train_v2_robust/val.parquet`) so **all** labeled mixed rows stay in train — no duplicate holdout. |
| 4 | **Pseudo:** `validator` rows, `cluster != -1`, `cluster_probability ≥` threshold; map cluster **0→human (0)**, **1→bot (1)** (override with `--cluster-human` / `--cluster-bot`). |
| 5 | **Weights:** real rows `sample_weight=1.0`; pseudo rows `pseudo_weight × cluster_probability`. |
| 6 | **Optional:** `--agreement logistic` keeps pseudo rows only if a quick logistic model (fit on labeled train split) agrees with the cluster label. |
| 7 | Write `train.parquet` (features + `label` + `sample_weight`) and `ssl_prepare_summary.json`. |

**Recommended** when mixed train already comes from the same feature pipeline: use your **fixed** `val.parquet` (all 8000 labeled mixed rows + pseudo in `train.parquet`):

```bash
PYTHONPATH=. python workspace/ssl_data/usl_hdbscan/human_bot_validator/prepare_weak_ssl_dataset.py \
  --input workspace/ssl_data/usl_hdbscan/human_bot_validator/mixed_train_with_clusters.parquet \
  --out-dir workspace/ssl_data/usl_hdbscan/human_bot_validator/ssl_weak_step1 \
  --val-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust/val.parquet \
  --min-cluster-prob 0.7 \
  --pseudo-weight 0.15 \
  --seed 42
```

**Alternative:** internal stratified split from labeled mixed only (ignores external val):

```bash
  ... --val-fraction 0.15
```
(omit `--val-parquet`).

**Second round (your idea):** run inference on medium-confidence rows outside this script, build a new `--input` or concat rows, and rerun `prepare_weak_ssl_dataset.py` with `--no-pseudo` plus your hand-built augment, or extend the script later. **Evaluate** using `val.parquet` from this step (real labels only).

**Train LGBM** on the prepared dir (`val` is unweighted; train uses weights):

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \
  --data-dir workspace/ssl_data/SSL/data/ssl_weak_step1 \
  --sample-weight-col sample_weight \
  --out-dir workspace/model/artifacts/lgbm_weak_ssl_step1
```

**Fair compare to `lgbm_feature56_v4_ultralow`** (same capacity / regularization as that artifact; only data + weights differ):

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \
  --data-dir workspace/ssl_data/SSL/data/ssl_weak_step1 \
  --sample-weight-col sample_weight \
  --out-dir workspace/model/artifacts/lgbm_weak_ssl_ultralow \
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

**Calibrate** the **raw** joblib (not required to retrain). Uses **`val.parquet`** from the same **`--data-dir`** you trained with (labeled holdout; no `sample_weight` column needed on val).

```bash
PYTHONPATH=. python workspace/model/scripts/calibrate_lgbm_joblib.py \
  --artifact-dir workspace/model/artifacts/lgbm_weak_ssl_ultralow \
  --data-dir workspace/ssl_data/SSL/data/ssl_weak_step1 \
  --method sigmoid \
  --report-threshold 0.5
```

Writes next to the artifact:

- `lgbm_b_classifier_calibrated.joblib`
- `lgbm_b_classifier_calibrated.calibration.json`

Point **`POKER44_MINER_MODEL_PATH`** (or your deploy path) at the **calibrated** joblib if you want miner `>= 0.5` to use calibrated probabilities.

If your prepared data lives under **`human_bot_validator/ssl_weak_step1`** instead, use that path for **`--data-dir`** here and in **`lgbm_2`** above.

Threshold sweep: among thresholds with max bot recall under target human FPR, the chosen cutoff is **closest to `--threshold-tie-ref`** (default **0.5**), then lower *t* if still tied — avoids useless extremes when val separates cleanly.

Ablations: `--no-pseudo` (labeled split only), `--pseudo-fraction 0.25`, `--agreement logistic`.

## 5) Outputs

- `mixed_train.parquet` — mixed rows + `mix_source` (`train_human` / `train_bot` / `validator`).
- `mixed_clusters.parquet` — `cluster` (+ optional `cluster_probability`); **row order matches `--input`** (join back to mixed train by index).
- `mixed_train_with_clusters.parquet` — wide table from `join_mixed_clusters.py` (train columns + cluster columns).
- `mixed_clusters.json` — run metadata (`scaled: true` when StandardScaler ran).
- `ssl_weak_step1/` (or your `--out-dir`) — `train.parquet`, `val.parquet`, `ssl_prepare_summary.json` from `prepare_weak_ssl_dataset.py`.
