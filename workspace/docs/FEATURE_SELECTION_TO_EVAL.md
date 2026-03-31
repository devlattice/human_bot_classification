# Feature Selection and Robust Dataset (Subnet Target)

This document is the **subnet** workflow for:

1) statistical screening with ANOVA/FDR/Bonferroni,  
2) robust feature transforms,  
3) LGBM-B training on subnet data.

It is aligned with current scripts under `workspace/preprocess`.

---

## 0) Inputs

Expected unprocessed dataset:

- `workspace/dataset/unpreprocessed/original_train/train.parquet`
- `workspace/dataset/unpreprocessed/original_train/val.parquet`

Feature lists used by transform (default):

- `workspace/preprocess/features/keep_features.txt`
- `workspace/preprocess/features/heavy_transform_features.txt`
- `workspace/preprocess/features/regularize_features.txt`

---

## 1) ANOVA + Bonferroni + FDR

Run on train+val of your subnet dataset:

```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/anova_bonferroni_FDR_test.py \
  --data-dir workspace/dataset/unpreprocessed/original_train \
  --disable-domain-shift-merge \
  --out-csv workspace/preprocess/statistical_test/original_train_anova.csv
```

Read these columns:

- `anova_f`, `p_value`
- `p_fdr_bh`
- `sig_bonferroni`
- `sig_fdr_0_05`
- `keep_score` (if domain-shift merge is enabled and available)

Notes:

- This script currently does **not** output `cohens_d`/`abs_cohens_d`.
- It currently does **not** support `--per-source`.

---

## 2) Auto-generate keep/watch/drop lists

Generate feature lists directly from the ANOVA CSV:

```bash
PYTHONPATH=. python workspace/preprocess/statistical_test/select_features.py \
  --anova-csv workspace/preprocess/statistical_test/original_train_anova.csv \
  --out-dir workspace/preprocess/feature_selection
```

Generated files:

- `workspace/preprocess/feature_selection/keep_features.txt`
- `workspace/preprocess/feature_selection/watch_features.txt`
- `workspace/preprocess/feature_selection/drop_features.txt`
- `workspace/preprocess/feature_selection/selection_summary.csv`

Robustness policy used by this script:

- `drop`: non-significant / degenerate / constant-like
- `watch`: significant but borderline
- `keep`: robust significant

---

## 3) Build robusted dataset

Use robust transform with stats fit from a reference parquet (recommended):

```bash
PYTHONPATH=. python workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/dataset/unpreprocessed/original_train \
  --out-dir workspace/dataset/robusted_dataset/original_train \
  --fit-stats-from workspace/dataset/unpreprocessed/original_train/train.parquet \
  --keep-features-file workspace/preprocess/feature_selection/keep_features.txt \
  --restrict-to-keep-features \
  --q-low 0.01 \
  --q-high 0.99 \
  --enable-log1p \
  --enable-robust-scale \
  --scaled-clip-abs 8.0 \
  --drop-row-nan-frac-over -1
```

Why:

- clipping + robust scaling reduce tail sensitivity and saturation
- `--fit-stats-from` keeps transform stats stable and reproducible
- `--restrict-to-keep-features` enforces the auto-selected keep subset

---

## 4) Train LGBM

```bash
PYTHONPATH=. python workspace/model/scripts/lgbm.py \
  --data-dir workspace/dataset/robusted_dataset/original_train \
  --out-dir workspace/model/artifacts/lgbm
```

---

## 5) Optional: drop aggressive features first

If you want a hard drop stage before robust transform:

```bash
PYTHONPATH=. python workspace/preprocess/drop_aggressive_features.py \
  --data-dir workspace/dataset/unpreprocessed/original_train \
  --out-dir workspace/dataset/unpreprocessed/original_train_drop \
  --features-file workspace/preprocess/features/heavy_transform_features.txt
```

Then run robust transform on `original_train_drop`.

---

## Minimal repeatable pipeline

```bash
# 1) stats
PYTHONPATH=. python workspace/preprocess/statistical_test/anova_bonferroni_FDR_test.py \
  --data-dir workspace/dataset/unpreprocessed/original_train \
  --disable-domain-shift-merge \
  --out-csv workspace/preprocess/statistical_test/original_train_anova.csv

# 2) auto-select keep/watch/drop
PYTHONPATH=. python workspace/preprocess/statistical_test/select_features.py \
  --anova-csv workspace/preprocess/statistical_test/original_train_anova.csv \
  --out-dir workspace/preprocess/feature_selection

# 3) robust transform
PYTHONPATH=. python workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/dataset/unpreprocessed/original_train \
  --out-dir workspace/dataset/robusted_dataset/original_train_selected \
  --fit-stats-from workspace/dataset/unpreprocessed/original_train/train.parquet \
  --keep-features-file workspace/preprocess/feature_selection/keep_features.txt \
  --restrict-to-keep-features \
  --q-low 0.01 --q-high 0.99 \
  --enable-log1p --enable-robust-scale --scaled-clip-abs 8.0 \
  --drop-row-nan-frac-over -1

# 4) train
PYTHONPATH=. python workspace/model/LGBM.py \
  --data-dir workspace/dataset/robusted_dataset/original_train_selected \
  --out-dir workspace/model/artifacts/lgbm_b_subnet_selected
```

