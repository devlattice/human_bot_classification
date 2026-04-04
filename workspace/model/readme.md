# Model training + calibration (56-feature robust path)

## Train `lgbm_2` — **v4 ultralow** (`lgbm_feature56_v4_ultralow`)

Stricter capacity than `lgbm_feature56_v3_lowcap_m1` (fewer leaves, shallower trees, larger `min_child_samples`) to reduce score saturation before adding pseudo-labeled validator rows.

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

**Prior run (reference):** `lgbm_feature56_v3_lowcap_m1` — `num_leaves=7`, `max_depth=3`, `min_child_samples=800`, etc. (see that folder’s `metrics.json`).

---

# Calibration (keep production threshold = 0.5)

Use post-hoc calibration to make predicted probabilities better aligned while
keeping the production rule `predict_bot = (proba >= 0.5)`.

## 1) Calibrate trained `lgbm_2` artifact (v4)

```bash
PYTHONPATH=. python workspace/model/scripts/calibrate_lgbm_joblib.py \
  --artifact-dir workspace/model/artifacts/lgbm_feature56_v4_ultralow \
  --data-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/train_v2_robust \
  --method sigmoid \
  --report-threshold 0.5
```

Outputs (under the artifact dir):
- `lgbm_b_classifier_calibrated.joblib`
- `lgbm_b_classifier_calibrated.calibration.json`

(To calibrate **v3** instead, set `--artifact-dir` to `workspace/model/artifacts/lgbm_feature56_v3_lowcap_m1`.)

Notes:
- `sigmoid` (Platt scaling) is the default and usually the safest first pass.
- If validation is large and diverse, try `--method isotonic` as a follow-up.

## 2) Cross-evaluate calibrated model at threshold 0.5

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_feature56_v4_ultralow/lgbm_b_classifier_calibrated.joblib \
  --metrics-json workspace/model/artifacts/lgbm_feature56_v4_ultralow/metrics.json \
  --threshold 0.5 \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature56_v4_ultralow_calibrated
```

### Cross-eval: which threshold?

`cross_dataset_eval.py` can report **three** notions on the same scored tables:

| Goal | Flags | What you read in `cross_dataset_comparison.csv` |
|------|--------|--------------------------------------------------|
| **Production @ 0.5** | `--threshold 0.5` + `--metrics-json …/metrics.json` | **`threshold_used`** and **`val_*`** (accuracy, human_fpr, …). Still get **`val_*_at_selected`** if metrics JSON has `selected_threshold` (often ~0.823 for v4 ultralow). |
| **Training sweep as primary** | Omit **`--threshold`**, keep **`--metrics-json`** | Primary **`threshold_used`** becomes **`threshold_selection.selected_threshold`** from that JSON (tie-break: closest to `--threshold-tie-ref`, default 0.5). For **`lgbm_weak_ssl_ultralow`** this may be **0.5**; for **`lgbm_feature56_v4_ultralow`** often **~0.823**. |
| **Per-dataset Youden *J*** | (default; use **`--no-youden`** to disable) | Second table in **`cross_dataset_comparison.md`**, or columns **`val_youden_threshold`**, **`val_accuracy_at_youden`**, **`val_human_fpr_at_youden`**, **`val_bot_recall_at_youden`** — optimal *t* on **that** eval parquet (max **TPR − FPR**; tie-break: higher *t*). |

**Weak-SSL ultralow (calibrated), same parquets — primary = training selected (no `--threshold`):**

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_weak_ssl_ultralow/lgbm_b_classifier_calibrated.joblib \
  --metrics-json workspace/model/artifacts/lgbm_weak_ssl_ultralow/metrics.json \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature56_ssl_calibrated_metrics_primary
```

**v4 ultralow calibrated — same, primary = training selected:**

```bash
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model workspace/model/artifacts/lgbm_feature56_v4_ultralow/lgbm_b_classifier_calibrated.joblib \
  --metrics-json workspace/model/artifacts/lgbm_feature56_v4_ultralow/metrics.json \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet \
  --eval-parquet workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet \
  --out-dir workspace/preprocess/statistical_test/explorer/miner_1/feature_1/cross_eval/feature56_v4_ultralow_calibrated_metrics_primary
```

On **shifted holdouts**, compare **Youden** rows to **0.5** and to **training *t***; none of the three is a substitute for **labeled validator** if that is the real target.

## 3) Quick checks after calibration

- Compare `brier_score_loss_calibrated` vs `brier_score_loss_raw` in calibration JSON.
- Compare `at_report_threshold_calibrated` vs `at_report_threshold_raw` (especially `human_fpr` and `bot_recall`).
- Compare calibrated vs uncalibrated cross-eval CSV on the same eval parquets.
