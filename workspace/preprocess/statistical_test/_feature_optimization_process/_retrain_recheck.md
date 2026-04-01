
# First modify keep_features from analysis of  train_vs_validator_robusted

```bash
python3 workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/dataset/unpreprocessed/train \
  --out-dir workspace/dataset/robusted_dataset/train \
  --fit-stats-from workspace/dataset/unpreprocessed/train/train.parquet \
  --keep-features-file workspace/preprocess/statistical_test/artifacts/feature_selection/keep_features.txt \
  --restrict-to-keep-features \
  --enable-log1p \
  --enable-robust-scale
```
e.g.
```bash
python3 workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/dataset/unpreprocessed/train \
  --out-dir workspace/dataset/robusted_dataset/train \
  --fit-stats-from workspace/dataset/unpreprocessed/train/train.parquet \
  --keep-features-file workspace/preprocess/statistical_test/artifacts/feature_selection/feature_tune/keep_features_v3b.txt \
  --restrict-to-keep-features \
  --enable-log1p \
  --enable-robust-scale
```
## Apply the training transform_meta.json to miner logs (same as inference):
```bash

cd /home/dr/Workspace/Poker44-subnet

python3 workspace/preprocess/robust_feature_transform.py \
  --transform-meta-in workspace/dataset/robusted_dataset/train/transform_meta.json \
  --in-parquet workspace/ssl_data/raw_data/validator_request.parquet \
  --out-parquet workspace/ssl_data/raw_data/validator_request_robusted.parquet

```
## KS + plots on robust train vs robust validator (shared 60 numeric features):
```bash
python3 workspace/preprocess/statistical_test/train_validator_shift_plots.py \
  --train-parquet workspace/dataset/robusted_dataset/train/train.parquet \
  --validator-parquet workspace/ssl_data/raw_data/validator_request_robusted.parquet \
  --out-dir workspace/preprocess/statistical_test/plots/train_vs_validator_robusted \
  --max-rows-per-source 0

```
Please repeat above process many times and finally acquire optimized features.

## Comparison summary (snapshots vs current)

Current file:
- `workspace/preprocess/statistical_test/plots/train_vs_validator_robusted/train_vs_validator_shift.csv`

Saved snapshots:
- `workspace/preprocess/statistical_test/_feature_optimization_process/train_vs_validator_shift.csv`
- `workspace/preprocess/statistical_test/_feature_optimization_process/train_vs_validator_shift_1.csv`
- `workspace/preprocess/statistical_test/_feature_optimization_process/train_vs_validator_shift_2.csv`

| file | rows | fdr_sig | ks>=0.30 | ks>=0.20 | ks>=0.15 | ks>=0.10 | ks_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| current `train_vs_validator_robusted/train_vs_validator_shift.csv` | 48 | 44 | 0 | 4 | 14 | 25 | 0.1067 |
| `train_vs_validator_shift.csv` (old baseline) | 79 | 62 | 3 | 17 | 29 | 43 | 0.1209 |
| `train_vs_validator_shift_1.csv` | 60 | 56 | 3 | 16 | 26 | 37 | 0.1408 |
| `train_vs_validator_shift_2.csv` | 52 | 48 | 0 | 8 | 18 | 29 | 0.1177 |

Interpretation:
- Overall shift metrics improved across iterations (especially severe-shift counts).
- Current set is best among saved snapshots by `ks>=0.20` and `ks>=0.30`.
- Domain shift is reduced but not eliminated (`fdr_sig=44/48`).