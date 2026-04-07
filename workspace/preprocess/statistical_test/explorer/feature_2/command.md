

Prune after `sanity_check` (use the **same** `--data-dir` you used for the report; edit `config/drop_after_sanity.txt` as needed):

```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/subset_parquet_columns.py \
  --data-dir <DIR_USED_FOR_SANITY> \
  --drop-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/drop_after_sanity.txt \
  --out-dir <DIR_PRUNED> \
  --write-keep-list workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/keep_features.txt

PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/sanity_check.py \
  --data-dir <DIR_PRUNED> \
  --out-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/sanity_report_pruned.csv \
  --out-json workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/sanity_summary_pruned.json
```
e.g.
```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/subset_parquet_columns.py \
  --data-dir workspace/dataset/unpreprocessed/train/system_bot \
  --drop-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/drop_after_sanity.txt \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train\
  --write-keep-list workspace/preprocess/statistical_test/explorer/miner_1/feature_1/config/keep_features.txt
```

Then ANOVA + `train_validator_shift_plots` on `<DIR_PRUNED>` (or equivalent parquets).

## Run anova_bonferroni_FDR_test.py on the same labeled matrix → per-feature F, p, Bonferroni, FDR.

```bash
python3 workspace/preprocess/statistical_test/anova_bonferroni_FDR_test.py \
  --parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train/train.parquet \
  --disable-domain-shift-merge \
  --out-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/task_signal/anova_bonferroni_FDR_combined.csv \
  --plots-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/task_signal/plots
```
## Domain shift (train vs validator)

### preparing validator dataset
```bash
python3 workspace/ssl_data/build_raw_dataset_for_domain.py \
  --input-source-dir workspace/ssl_data/json \
  --sample workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train/train.parquet \
  --outdir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/validator \
  --output-name request.parquet
```
### Run train_validator_shift_plots.py (train parquet vs validator parquet, same feature set) → train_vs_validator_shift.csv + plots.

```bash
cd /path/to/Poker44-subnet

python3 workspace/preprocess/statistical_test/train_validator_shift_plots.py \
  --train-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train/train.parquet \
  --validator-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/validator/request.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/shift \
  --max-rows-per-source 0
```

### Merge and decide
```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/primary_report.py \
  --anova-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/task_signal/anova_bonferroni_FDR_combined.csv \
  --shift-csv workspace/preprocess/statistical_test/explorer/miner_1/feature_1/shift/train_vs_validator_shift.csv \
  --output workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection
```
### Subset to 62 features (`keep` ∪ `keep_watch_shift`)

Use **`feature_manually.txt`** (or concat the two `features_*.txt` lists). Requires **`--keep-file`** (not `--drop-file`).

```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/subset_parquet_columns.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train \
  --keep-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection/feature_manually.txt \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62
```
Optional: same **`--keep-file`** with **`--parquet IN OUT`** for each test parquet you will score.

### Robust transform: fit on train, write train/val + `transform_meta.json`

Fit stats from **`train.parquet` only** (same flags as `workspace/docs/FEATURE_SELECTION_TO_EVAL.md`). **Do not** reuse a `transform_meta.json` fit on a wider feature set; refit after locking the 62 columns.

```bash
PYTHONPATH=. python workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62 \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62_robust \
  --fit-stats-from workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62/train.parquet \
  --keep-features-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection/feature_manually.txt \
  --restrict-to-keep-features \
  --q-low 0.01 --q-high 0.99 \
  --enable-log1p --enable-robust-scale --scaled-clip-abs 8.0 \
  --drop-row-nan-frac-over -1
```

Meta path for apply-only: **`.../data/train_62_robust/transform_meta.json`**.

### Preparing test / validator parquets (apply-only)

Subset each file to the **same 62** columns first (see **`--parquet IN OUT`** above), then apply the meta:

```bash
META=workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62_robust/transform_meta.json

PYTHONPATH=. python workspace/preprocess/robust_feature_transform.py \
  --transform-meta-in "$META" \
  --in-parquet <PATH_TO_TEST_62COL.parquet> \
  --out-parquet <PATH_TO_TEST_62COL_ROBUST.parquet>
```

Repeat **`--in-parquet` / `--out-parquet`** per file.

### Train LightGBM (all non-label columns = features)

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_62_robust \
  --out-dir workspace/model/artifacts/lgbm_feature_62_m1 \
  --device cpu
```

Artifact: **`lgbm_b_classifier.joblib`** (not `lgbm_classifier.joblib`).

### Cross-eval

Point **`--eval-parquet`** at test files that are **subset to the 62** and **transformed with the same `transform_meta.json`**.

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_feature_62_m1/lgbm_b_classifier.joblib \
  --metrics-json workspace/model/artifacts/lgbm_feature_62_m1/metrics.json \
  --threshold 0.5 \
  --eval-parquet <PATH_TO_TEST_62COL_ROBUST.parquet> \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval
```
e.g.
```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_feature_62_m1/lgbm_b_classifier.joblib \
  --metrics-json workspace/model/artifacts/lgbm_feature_62_m1/metrics.json \
  --threshold 0.5 \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature/feature62
```

## Ablation: v2 shift drop (**56** features)

**Dropped** (high train↔validator KS and/or weak `task_importance_score` in ANOVA): see `features_selection/features_drop_shift_v2.txt` — `fold_ratio_max`, `p3_max`, `p3_mean`, `p3_std`, `p6p_std`, `n_players_mean`.

**Keep list:** `features_selection/feature_manually_v2_drop_shift.txt`. **Refit** `transform_meta.json` on this set (do not reuse the 62-col meta).

### Subset train → `train_v2`

```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/explorer/miner_1/feature_1/scripts/subset_parquet_columns.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train \
  --keep-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection/feature_manually_v2_drop_shift.txt \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2
```

### Robust transform → `train_v2_robust`

```bash
PYTHONPATH=. python workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2 \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust \
  --fit-stats-from workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2/train.parquet \
  --keep-features-file workspace/preprocess/statistical_test/explorer/miner_1/feature_1/features_selection/feature_manually_v2_drop_shift.txt \
  --restrict-to-keep-features \
  --q-low 0.01 --q-high 0.99 \
  --enable-log1p --enable-robust-scale --scaled-clip-abs 8.0 \
  --drop-row-nan-frac-over -1
```

**Meta:** `.../data/train_v2_robust/transform_meta.json`. Subset each test parquet with **`feature_manually_v2_drop_shift.txt`**, then apply-only with this meta (same pattern as above).

### Train + cross-eval (separate artifact)

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust \
  --out-dir workspace/model/artifacts/lgbm_feature_v2_shiftdrop_m1 \
  --device cpu
```

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_feature_v2_shiftdrop_m1/lgbm_b_classifier.joblib \
  --metrics-json workspace/model/artifacts/lgbm_feature_v2_shiftdrop_m1/metrics.json \
  --threshold 0.5 \
  --eval-parquet <PATH_TO_TEST_56COL_ROBUST.parquet> \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature56
```

### Train `lgbm_2` ultralow → `lgbm_feature56_v4_ultralow`

Lower capacity than `lgbm_feature56_v3_lowcap_m1` (see `workspace/model/readme.md` for v3 vs v4 params). Same **`train_v2_robust`** data.

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust \
  --out-dir workspace/model/artifacts/lgbm_feature56_v4_ultralow \
  --device cpu \
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

Compare **`cross_eval`** (62-col / `lgbm_feature_62_m1`) vs **`cross_eval_v2_shiftdrop`** on the **same** test files (each run through its matching keep-list + meta).