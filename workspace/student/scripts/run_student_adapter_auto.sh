#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

# Default feature list aligned with explorer feature_3 v1 (same rb_B pipeline as grid / holdouts).
FEATURES_TXT="${FEATURES_TXT:-workspace/preprocess/statistical_test/explorer/feature_3/config/v1/keep_features.txt}"
SRC_TRAIN="${SRC_TRAIN:-workspace/preprocess/statistical_test/explorer/feature_3/data/train/public/rb_B/train.parquet}"
SRC_VAL="${SRC_VAL:-workspace/preprocess/statistical_test/explorer/feature_3/data/train/public/rb_B/val.parquet}"
UNLABELED_VALIDATOR="${UNLABELED_VALIDATOR:-workspace/preprocess/statistical_test/explorer/feature_3/data/validator/rb_B/validator.parquet}"

HOLDOUT_1="${HOLDOUT_1:-workspace/preprocess/statistical_test/explorer/feature_3/data/test/holdout/rb_B/holdout_1.parquet}"
HOLDOUT_2="${HOLDOUT_2:-workspace/preprocess/statistical_test/explorer/feature_3/data/test/holdout/rb_B/holdout_2.parquet}"
HOLDOUT_PARQUETS=()
TUNE_PARQUETS=()
TEST_PARQUETS=()

ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-80}"
ADAPTER_WARMUP_EPOCHS="${ADAPTER_WARMUP_EPOCHS:-8}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-256}"
# DL adapter defaults from adapter_grid_domain_conf_v2 grid_summary best_trial (trial 7).
ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM:-128}"
ADAPTER_EMBED_DIM="${ADAPTER_EMBED_DIM:-32}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.12}"
ADAPTER_LR="${ADAPTER_LR:-0.0003}"
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-0.0001}"
ADAPTER_LAMBDA_DOMAIN_MAX="${ADAPTER_LAMBDA_DOMAIN_MAX:-0.05}"
ADAPTER_LAMBDA_DOMAIN_GAMMA="${ADAPTER_LAMBDA_DOMAIN_GAMMA:-8.0}"
ADAPTER_DOMAIN_SELECTION_WEIGHT="${ADAPTER_DOMAIN_SELECTION_WEIGHT:-0.6}"
ADAPTER_DOMAIN_EVAL_TARGET_ROWS="${ADAPTER_DOMAIN_EVAL_TARGET_ROWS:-10000}"

OPTUNA_TRIALS="${OPTUNA_TRIALS:-60}"
OPTUNA_SAMPLER="${OPTUNA_SAMPLER:-tpe}"
OPTUNA_OBJECTIVE="${OPTUNA_OBJECTIVE:-multi_objective_generalization_at_05}"
OPTUNA_FPR_CAP="${OPTUNA_FPR_CAP:-0.05}"
OPTUNA_HOLDOUT_FPR_CAP="${OPTUNA_HOLDOUT_FPR_CAP:-0.08}"
OPTUNA_HOLDOUT_PENALTY="${OPTUNA_HOLDOUT_PENALTY:-0.5}"
LGBM_DEVICE="${LGBM_DEVICE:-cpu}"
LGBM_REGULARIZATION="${LGBM_REGULARIZATION:-strong}"

OUT_DIR="${OUT_DIR:-workspace/student/artifacts/student_adapter_auto_$(date +%Y%m%d_%H%M%S)}"
CONCAT_ORIGINAL="${CONCAT_ORIGINAL:-1}"

usage() {
  cat <<'USAGE'
Usage: bash workspace/student/scripts/run_student_adapter_auto.sh [options]

Options:
  --out-dir DIR
  --features-txt PATH
  --src-train PATH
  --src-val PATH
  --unlabeled-validator PATH
  --holdout-1 PATH
  --holdout-2 PATH
  --holdout-parquet PATH      Repeatable additional test parquet (legacy alias).
  --tune-parquet PATH         Repeatable tune parquet for optimization (default: holdout-1/2).
  --test-parquet PATH         Repeatable test parquet for final reporting.
  --seed INT
  --device auto|cpu|cuda
  --lgbm-device cpu|gpu
  --optuna-trials INT
  --optuna-objective bot_recall_at_05|max_bot_recall_at_fpr|multi_objective_generalization_at_05
  --adapter-domain-selection-weight FLOAT
  --adapter-domain-eval-target-rows INT
  --no-concat-original          Export adapter embeddings only (no raw features).
  -h, --help

Environment variables are also supported for all knobs in this script.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --features-txt) FEATURES_TXT="${2:?}"; shift 2 ;;
    --src-train) SRC_TRAIN="${2:?}"; shift 2 ;;
    --src-val) SRC_VAL="${2:?}"; shift 2 ;;
    --unlabeled-validator) UNLABELED_VALIDATOR="${2:?}"; shift 2 ;;
    --holdout-1) HOLDOUT_1="${2:?}"; shift 2 ;;
    --holdout-2) HOLDOUT_2="${2:?}"; shift 2 ;;
    --holdout-parquet) HOLDOUT_PARQUETS+=("${2:?}"); shift 2 ;;
    --tune-parquet) TUNE_PARQUETS+=("${2:?}"); shift 2 ;;
    --test-parquet) TEST_PARQUETS+=("${2:?}"); shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --lgbm-device) LGBM_DEVICE="${2:?}"; shift 2 ;;
    --optuna-trials) OPTUNA_TRIALS="${2:?}"; shift 2 ;;
    --optuna-objective) OPTUNA_OBJECTIVE="${2:?}"; shift 2 ;;
    --adapter-domain-selection-weight) ADAPTER_DOMAIN_SELECTION_WEIGHT="${2:?}"; shift 2 ;;
    --adapter-domain-eval-target-rows) ADAPTER_DOMAIN_EVAL_TARGET_ROWS="${2:?}"; shift 2 ;;
    --no-concat-original) CONCAT_ORIGINAL="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for _f in "${FEATURES_TXT}" "${SRC_TRAIN}" "${SRC_VAL}" "${UNLABELED_VALIDATOR}" "${HOLDOUT_1}" "${HOLDOUT_2}"; do
  [[ -f "${_f}" ]] || { echo "Missing required file: ${_f}" >&2; exit 1; }
done
for _f in "${HOLDOUT_PARQUETS[@]}" "${TUNE_PARQUETS[@]}" "${TEST_PARQUETS[@]}"; do
  [[ -f "${_f}" ]] || { echo "Missing required file: ${_f}" >&2; exit 1; }
done

if [[ ${#TUNE_PARQUETS[@]} -eq 0 ]]; then
  TUNE_PARQUETS=("${HOLDOUT_1}" "${HOLDOUT_2}")
fi
if [[ ${#TEST_PARQUETS[@]} -eq 0 ]]; then
  TEST_PARQUETS=("${HOLDOUT_PARQUETS[@]}")
fi

ALL_HOLDOUTS=("${TUNE_PARQUETS[@]}" "${TEST_PARQUETS[@]}")
declare -A _BASENAME_COUNT=()
for _h in "${ALL_HOLDOUTS[@]}"; do
  _bn="$(basename "${_h}")"
  _BASENAME_COUNT["${_bn}"]=$(( ${_BASENAME_COUNT["${_bn}"]:-0} + 1 ))
done
for _k in "${!_BASENAME_COUNT[@]}"; do
  if [[ ${_BASENAME_COUNT["${_k}"]} -gt 1 ]]; then
    echo "Duplicate parquet basename detected across tune/test inputs: ${_k}" >&2
    echo "Please rename files or avoid basename collisions before running auto pipeline." >&2
    exit 1
  fi
done

EMB_HOLDOUT_ARGS=()
for _h in "${ALL_HOLDOUTS[@]}"; do
  EMB_HOLDOUT_ARGS+=(--in-parquet "${_h}")
done

OUT_DIR="$(python3 - <<PY
from pathlib import Path
print(Path("${OUT_DIR}").expanduser().resolve())
PY
)"
mkdir -p "${OUT_DIR}"

echo "[student-auto] out_dir=${OUT_DIR}"
echo "[student-auto] step 1/4 train DL adapter"
PYTHONPATH=. python workspace/student/scripts/train_dl_adapter.py \
  --source-train "${SRC_TRAIN}" \
  --source-val "${SRC_VAL}" \
  --target-unlabeled "${UNLABELED_VALIDATOR}" \
  --feature-cols-file "${FEATURES_TXT}" \
  --out-dir "${OUT_DIR}/adapter" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --epochs "${ADAPTER_EPOCHS}" \
  --warmup-epochs "${ADAPTER_WARMUP_EPOCHS}" \
  --batch-size "${ADAPTER_BATCH_SIZE}" \
  --hidden-dim "${ADAPTER_HIDDEN_DIM}" \
  --embed-dim "${ADAPTER_EMBED_DIM}" \
  --dropout "${ADAPTER_DROPOUT}" \
  --lr "${ADAPTER_LR}" \
  --weight-decay "${ADAPTER_WEIGHT_DECAY}" \
  --lambda-domain-max "${ADAPTER_LAMBDA_DOMAIN_MAX}" \
  --lambda-domain-gamma "${ADAPTER_LAMBDA_DOMAIN_GAMMA}" \
  --domain-selection-weight "${ADAPTER_DOMAIN_SELECTION_WEIGHT}" \
  --domain-eval-target-rows "${ADAPTER_DOMAIN_EVAL_TARGET_ROWS}"

echo "[student-auto] step 2/4 export adapter embeddings"
EMB_ARGS=()
if [[ "${CONCAT_ORIGINAL}" == "1" ]]; then
  EMB_ARGS+=(--concat-original-features)
fi
PYTHONPATH=. python workspace/student/scripts/export_adapter_embeddings.py \
  --artifact "${OUT_DIR}/adapter/dl_adapter.pt" \
  --in-parquet "${SRC_TRAIN}" \
  --in-parquet "${SRC_VAL}" \
  --in-parquet "${UNLABELED_VALIDATOR}" \
  "${EMB_HOLDOUT_ARGS[@]}" \
  --out-dir "${OUT_DIR}/embeddings" \
  --device "${DEVICE}" \
  "${EMB_ARGS[@]}"

echo "[student-auto] step 3/4 prepare lgbm data"
PYTHONPATH=. python workspace/student/scripts/prepare_student_lgbm_data.py \
  --train "${OUT_DIR}/embeddings/$(basename "${SRC_TRAIN}")" \
  --val "${OUT_DIR}/embeddings/$(basename "${SRC_VAL}")" \
  --out-dir "${OUT_DIR}/lgbm_data"

TUNE_ARGS=()
for _h in "${TUNE_PARQUETS[@]}"; do
  TUNE_ARGS+=(--tune-parquet "${OUT_DIR}/embeddings/$(basename "${_h}")")
done
TEST_ARGS=()
for _h in "${TEST_PARQUETS[@]}"; do
  TEST_ARGS+=(--test-parquet "${OUT_DIR}/embeddings/$(basename "${_h}")")
done

echo "[student-auto] step 4/4 tune student lgbm"
PYTHONPATH=. python workspace/student/scripts/tune_student_lgbm_optuna.py \
  --data-dir "${OUT_DIR}/lgbm_data" \
  --out-dir "${OUT_DIR}/lgbm_optuna" \
  --n-trials "${OPTUNA_TRIALS}" \
  --sampler "${OPTUNA_SAMPLER}" \
  --seed "${SEED}" \
  --lgbm-device "${LGBM_DEVICE}" \
  --objective "${OPTUNA_OBJECTIVE}" \
  --fpr-cap "${OPTUNA_FPR_CAP}" \
  "${TUNE_ARGS[@]}" \
  "${TEST_ARGS[@]}" \
  --holdout-fpr-cap "${OPTUNA_HOLDOUT_FPR_CAP}" \
  --holdout-penalty "${OPTUNA_HOLDOUT_PENALTY}" \
  --lgbm-regularization "${LGBM_REGULARIZATION}"

echo "[student-auto] done"
echo "[student-auto] adapter metrics: ${OUT_DIR}/adapter/adapter_metrics.json"
echo "[student-auto] lgbm summary:   ${OUT_DIR}/lgbm_optuna/optuna_summary.json"

