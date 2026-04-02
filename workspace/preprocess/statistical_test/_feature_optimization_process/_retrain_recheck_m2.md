
# First modify keep_features from analysis of  train_vs_validator_robusted

Try this mild-strong setting first (safer than aggressive clipping):
--q-low 0.02
--q-high 0.98
--scaled-clip-abs 6.0

If you want one more robust (second step):
--q-low 0.03 --q-high 0.97 --scaled-clip-abs 5.0

```bash
python3 workspace/preprocess/robust_feature_transform.py \
  --data-dir workspace/dataset/unpreprocessed/train/system_human_bot \
  --out-dir workspace/dataset/robusted_dataset/train/system_human_bot \
  --fit-stats-from workspace/dataset/unpreprocessed/train/system_human_bot/train.parquet \
  --keep-features-file workspace/preprocess/statistical_test/artifacts/feature_selection_m_2/feature_tune/keep_features_v3b.txt \
  --restrict-to-keep-features \
  --enable-log1p \
  --q-low 0.02 \
  --q-high 0.98 \
  --scaled-clip-abs 6.0 \
  --enable-robust-scale
```
## Apply the training transform_meta.json to miner logs (same as inference):
```bash

cd /home/dr/Workspace/Poker44-subnet

python3 workspace/preprocess/robust_feature_transform.py \
  --transform-meta-in workspace/dataset/robusted_dataset/train/system_human_bot/transform_meta.json \
  --in-parquet workspace/ssl_data/raw_data/miner_1/validator_request.parquet \
  --out-parquet workspace/ssl_data/raw_data/miner_1/validator_request_robusted.parquet

```
## KS + plots on robust train vs robust validator (shared 60 numeric features):
```bash
python3 workspace/preprocess/statistical_test/train_validator_shift_plots.py \
  --train-parquet workspace/dataset/robusted_dataset/train/system_human_bot/train.parquet \
  --validator-parquet workspace/ssl_data/raw_data/miner_1/validator_request_robusted.parquet \
  --out-dir workspace/preprocess/statistical_test/plots/miner_2/train_vs_validator_robusted \
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