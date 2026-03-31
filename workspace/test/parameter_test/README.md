# Parameter mini-grid (a, b, gamma)

This folder is a quick harness to compare the miner uncertain-band smoothing configs:

1. `(a,b,gamma) = (0.3, 0.7, 1.4)` (current)
2. `(a,b,gamma) = (0.4, 0.6, 1.4)`
3. `smoothing OFF` (invalid `a/b`, e.g. `-1/-1`, gamma ignored)

## Important scope note

`POKER44_MINER_UNCERTAIN_A/B/GAMMA` affects **runtime miner scores** in `neurons/miner.py`.
It does **not** change model training weights and does **not** affect standalone model scripts unless those scripts use miner runtime outputs.

So this mini-grid is for **inference policy behavior** (FPR/recall tradeoff), not retraining.

## Run plan (manual, repeat for each config)

From repo root:

### 1) Set env for one config and restart miner

```bash
# Example run A
export POKER44_MINER_UNCERTAIN_A=0.3
export POKER44_MINER_UNCERTAIN_B=0.7
export POKER44_MINER_UNCERTAIN_GAMMA=1.4

# restart your miner process (PM2 / run script as you already use)
bash scripts/miner/run/run_miner.sh
```

For run B:

```bash
export POKER44_MINER_UNCERTAIN_A=0.4
export POKER44_MINER_UNCERTAIN_B=0.6
export POKER44_MINER_UNCERTAIN_GAMMA=1.4
bash scripts/miner/run/run_miner.sh
```

For run C (smoothing OFF):

```bash
export POKER44_MINER_UNCERTAIN_A=-1
export POKER44_MINER_UNCERTAIN_B=-1
export POKER44_MINER_UNCERTAIN_GAMMA=1.0
bash scripts/miner/run/run_miner.sh
```

### 2) Evaluate and save per-run cross-eval CSV

Use your existing evaluator; this command is the standard one for model artifacts:

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_2_v1/lgbm_classifier.joblib \
  --eval-parquet workspace/dataset/robusted_dataset/original_test/hollout_human_mix.parquet \
  --eval-parquet workspace/dataset/robusted_dataset/original_test/hollout_train.parquet \
  --eval-parquet workspace/dataset/robusted_dataset/original_test/hollout_test.parquet \
  --threshold 0.6 \
  --selected-threshold 0.952 \
  --out-dir workspace/test/parameter_test/run_A
```

Repeat with `run_B`, `run_C` output dirs for each config.

## What to compare (primary)

For each dataset row (`hollout_human_mix`, `hollout_train`, `hollout_test`), compare:

- `val_human_fpr`
- `val_bot_recall`
- `val_accuracy`
- `val_human_fpr_at_selected`
- `val_bot_recall_at_selected`

Use:

- `results_template.md` for human-readable summary
- `results_template.csv` for spreadsheet copy/paste

## Decision rule (recommended)

Pick the config with:

1. Lowest `val_human_fpr` on shifted subsets (`hollout_train/test`), and
2. Highest `val_bot_recall` under your safety constraint.

If two configs are close, prefer the simpler / more conservative one.
