#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

USE_DBF="${USE_DBF:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
MANIFEST_SEARCH_DIR="${MANIFEST_SEARCH_DIR:-workspace/ssl_data/ssl_embed/artifacts}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-}"
RUN_OPTUNA="${RUN_OPTUNA:-0}"
OPTUNA_WORK_DIR="${OPTUNA_WORK_DIR:-}"
OPTUNA_OUTER_TRIALS="${OPTUNA_OUTER_TRIALS:-5}"
OPTUNA_INNER_TRIALS="${OPTUNA_INNER_TRIALS:-15}"
OPTUNA_OBJECTIVE="${OPTUNA_OBJECTIVE:-max_bot_recall_at_fpr}"
OPTUNA_FPR_CAP="${OPTUNA_FPR_CAP:-0.05}"
OPTUNA_EXTRA_ARGS=()
FEATURES_TXT_OVERRIDE="${FEATURES_TXT_OVERRIDE:-}"
SRC_TRAIN_OVERRIDE="${SRC_TRAIN_OVERRIDE:-}"
SRC_VAL_OVERRIDE="${SRC_VAL_OVERRIDE:-}"
UNLABELED_VALIDATOR_OVERRIDE="${UNLABELED_VALIDATOR_OVERRIDE:-}"
TEST_PARQUETS=()
QUANTILE_BOUNDS_JSON_OVERRIDE="${QUANTILE_BOUNDS_JSON_OVERRIDE:-}"

usage() {
  cat <<'USAGE'
Usage: bash workspace/ssl_data/ssl_embed/scripts/run_ssl_embed_auto.sh [options] [-- passthrough args]

Options:
  --use-dbf                  Prepare/use DBF inputs (equivalent to USE_DBF_INPUTS=1).
  --manifest PATH            Explicit manifest path; skips auto-discovery.
  --manifest-search-dir DIR  Where to search for latest ssl_embed_ablation_manifest.json.
  --device DEV               Compute device override: cuda|gpu or cpu.
  --cuda                     Shortcut for --device cuda.
  --cpu                      Shortcut for --device cpu.
  --run-optuna               Run tune_ssl_lgbm_optuna.py (ssl_lgbm) first, then use its manifest.
  --optuna-work-dir DIR      Work dir for Optuna run (default: auto under artifacts/).
  --optuna-outer-trials N    Outer SSL trials (default: 5).
  --optuna-inner-trials N    Inner LGBM trials (default: 15).
  --optuna-objective OBJ     bot_recall_at_05 | max_bot_recall_at_fpr (default: max_bot_recall_at_fpr).
  --optuna-fpr-cap V         Target human FPR cap for Optuna objective (default: 0.05).
  --optuna-arg ARG           Repeatable extra arg forwarded to tune_ssl_lgbm_optuna.py.
  --features-txt PATH        keep_features file to use.
  --src-train PATH           Source train parquet.
  --src-val PATH             Source val parquet (optional).
  --unlabeled-validator PATH Unlabeled validator parquet for SSL pool.
  --test-parquet PATH        Repeatable test parquet override (maps to TEST_PUB1..5).
  --quantile-bounds-json PATH
                             Train-fitted DBF winsor bounds JSON (default: feature_2/config/dbf_quantile_bounds.json).
  -h, --help                 Show this help.

Environment variables:
  USE_DBF=1                same as --use-dbf
  MANIFEST_PATH=...        same as --manifest
  MANIFEST_SEARCH_DIR=...  same as --manifest-search-dir
  LGBM_DEVICE=cpu|gpu      same target used by --device mapping below
  RUN_OPTUNA=1             same as --run-optuna
  OPTUNA_WORK_DIR=...      same as --optuna-work-dir
  FEATURES_TXT=...         same as --features-txt
  SRC_TRAIN=...            same as --src-train
  SRC_VAL=...              same as --src-val
  UNLABELED_VALIDATOR=...  same as --unlabeled-validator
  QUANTILE_BOUNDS_JSON=... same as --quantile-bounds-json

Passthrough:
  Any args after `--` are passed to run_ssl_lgbm_ablation.sh.
USAGE
}

PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-dbf)
      USE_DBF=1
      shift
      ;;
    --manifest)
      MANIFEST_PATH="${2:?--manifest requires a path}"
      shift 2
      ;;
    --manifest-search-dir)
      MANIFEST_SEARCH_DIR="${2:?--manifest-search-dir requires a directory}"
      shift 2
      ;;
    --device)
      DEVICE_OVERRIDE="${2:?--device requires cuda|gpu|cpu}"
      shift 2
      ;;
    --cuda)
      DEVICE_OVERRIDE="cuda"
      shift
      ;;
    --cpu)
      DEVICE_OVERRIDE="cpu"
      shift
      ;;
    --run-optuna)
      RUN_OPTUNA=1
      shift
      ;;
    --optuna-work-dir)
      OPTUNA_WORK_DIR="${2:?--optuna-work-dir requires a directory}"
      shift 2
      ;;
    --optuna-outer-trials)
      OPTUNA_OUTER_TRIALS="${2:?--optuna-outer-trials requires an integer}"
      shift 2
      ;;
    --optuna-inner-trials)
      OPTUNA_INNER_TRIALS="${2:?--optuna-inner-trials requires an integer}"
      shift 2
      ;;
    --optuna-objective)
      OPTUNA_OBJECTIVE="${2:?--optuna-objective requires a value}"
      shift 2
      ;;
    --optuna-fpr-cap)
      OPTUNA_FPR_CAP="${2:?--optuna-fpr-cap requires a float}"
      shift 2
      ;;
    --optuna-arg)
      OPTUNA_EXTRA_ARGS+=("${2:?--optuna-arg requires a value}")
      shift 2
      ;;
    --features-txt)
      FEATURES_TXT_OVERRIDE="${2:?--features-txt requires a path}"
      shift 2
      ;;
    --src-train)
      SRC_TRAIN_OVERRIDE="${2:?--src-train requires a path}"
      shift 2
      ;;
    --src-val)
      SRC_VAL_OVERRIDE="${2:?--src-val requires a path}"
      shift 2
      ;;
    --unlabeled-validator)
      UNLABELED_VALIDATOR_OVERRIDE="${2:?--unlabeled-validator requires a path}"
      shift 2
      ;;
    --test-parquet)
      TEST_PARQUETS+=("${2:?--test-parquet requires a path}")
      shift 2
      ;;
    --quantile-bounds-json)
      QUANTILE_BOUNDS_JSON_OVERRIDE="${2:?--quantile-bounds-json requires a path}"
      shift 2
      ;;
    --)
      shift
      PASSTHROUGH_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

QB_JSON_DEFAULT="workspace/preprocess/statistical_test/explorer/feature_2/config/dbf_quantile_bounds.json"
QB_JSON="${QUANTILE_BOUNDS_JSON_OVERRIDE:-${QUANTILE_BOUNDS_JSON:-${QB_JSON_DEFAULT}}}"

with_dbf_path() {
  local p="$1"
  if [[ "$p" == *_with_dbf.parquet ]]; then
    echo "$p"
    return
  fi
  if [[ "$p" == *.parquet ]]; then
    echo "${p%.parquet}_with_dbf.parquet"
    return
  fi
  echo "$p"
}

with_dbf_features_txt() {
  local p="$1"
  if [[ "$p" == *_with_dbf.txt ]]; then
    echo "$p"
    return
  fi
  if [[ "$p" == *.txt ]]; then
    echo "${p%.txt}_with_dbf.txt"
    return
  fi
  echo "$p"
}

if [[ -n "${FEATURES_TXT_OVERRIDE}" ]]; then
  export FEATURES_TXT="${FEATURES_TXT_OVERRIDE}"
fi
if [[ -n "${SRC_TRAIN_OVERRIDE}" ]]; then
  export SRC_TRAIN="${SRC_TRAIN_OVERRIDE}"
fi
if [[ -n "${SRC_VAL_OVERRIDE}" ]]; then
  export SRC_VAL="${SRC_VAL_OVERRIDE}"
fi
if [[ -n "${UNLABELED_VALIDATOR_OVERRIDE}" ]]; then
  export UNLABELED_VALIDATOR="${UNLABELED_VALIDATOR_OVERRIDE}"
fi

if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  case "${DEVICE_OVERRIDE}" in
    cuda|gpu)
      export LGBM_DEVICE="gpu"
      ;;
    cpu)
      export LGBM_DEVICE="cpu"
      ;;
    *)
      echo "Invalid --device value: ${DEVICE_OVERRIDE} (use cuda|gpu|cpu)" >&2
      exit 1
      ;;
  esac
fi

if [[ "${USE_DBF}" == "1" ]]; then
  echo "[auto] preparing DBF inputs"
  BASE_FEATURES_TXT="${FEATURES_TXT:-workspace/preprocess/statistical_test/explorer/feature_2/config/keep_features.txt}"
  OUT_FEATURES_TXT="$(with_dbf_features_txt "${BASE_FEATURES_TXT}")"
  PYTHONPATH=. python workspace/teacher/scripts/prepare_ssl_embed_dbf_inputs.py \
    --base-features-txt "${BASE_FEATURES_TXT}" \
    --out-features-txt "${OUT_FEATURES_TXT}" \
    --quantile-bounds-json "${QB_JSON}"
  export FEATURES_TXT="${OUT_FEATURES_TXT}"

  _dbf_qb_args=()
  if [[ -f "${QB_JSON}" ]]; then
    _dbf_qb_args=(--quantile-bounds-json "${QB_JSON}")
  else
    echo "[auto] warning: DBF quantile bounds missing at ${QB_JSON}; dbf.py will use per-split quantiles (leakage)." >&2
  fi

  # If users provided explicit parquets, make/use companion *_with_dbf.parquet files.
  for _var_name in SRC_TRAIN SRC_VAL UNLABELED_VALIDATOR; do
    _v="${!_var_name:-}"
    if [[ -n "${_v}" ]]; then
      _dbf="$(with_dbf_path "${_v}")"
      if [[ ! -f "${_dbf}" ]]; then
        echo "[auto] building DBF parquet: ${_dbf}"
        PYTHONPATH=. python workspace/teacher/scripts/dbf.py \
          --input-parquet "${_v}" \
          --output-parquet "${_dbf}" \
          "${_dbf_qb_args[@]}"
      fi
      export "${_var_name}=${_dbf}"
    fi
  done

  if [[ ${#TEST_PARQUETS[@]} -gt 0 ]]; then
    DBF_TESTS=()
    for _t in "${TEST_PARQUETS[@]}"; do
      _dbf_t="$(with_dbf_path "${_t}")"
      if [[ ! -f "${_dbf_t}" ]]; then
        echo "[auto] building DBF test parquet: ${_dbf_t}"
        PYTHONPATH=. python workspace/teacher/scripts/dbf.py \
          --input-parquet "${_t}" \
          --output-parquet "${_dbf_t}" \
          "${_dbf_qb_args[@]}"
      fi
      DBF_TESTS+=("${_dbf_t}")
    done
    TEST_PARQUETS=("${DBF_TESTS[@]}")
  fi
  export USE_DBF_INPUTS=1
fi

if [[ ${#TEST_PARQUETS[@]} -gt 0 ]]; then
  if [[ ${#TEST_PARQUETS[@]} -gt 5 ]]; then
    echo "[auto] warning: run_ssl_lgbm_ablation supports up to 5 tests; extra entries are ignored." >&2
  fi
  [[ ${#TEST_PARQUETS[@]} -ge 1 ]] && export TEST_PUB1="${TEST_PARQUETS[0]}"
  [[ ${#TEST_PARQUETS[@]} -ge 2 ]] && export TEST_PUB2="${TEST_PARQUETS[1]}"
  [[ ${#TEST_PARQUETS[@]} -ge 3 ]] && export TEST_PUB3="${TEST_PARQUETS[2]}"
  [[ ${#TEST_PARQUETS[@]} -ge 4 ]] && export TEST_PUB4="${TEST_PARQUETS[3]}"
  [[ ${#TEST_PARQUETS[@]} -ge 5 ]] && export TEST_PUB5="${TEST_PARQUETS[4]}"
fi

if [[ "${RUN_OPTUNA}" == "1" ]]; then
  _opt_features="${FEATURES_TXT:-workspace/preprocess/statistical_test/explorer/feature_2/config/keep_features.txt}"
  _opt_train="${SRC_TRAIN:-workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet}"
  _opt_val="${SRC_VAL:-workspace/preprocess/statistical_test/explorer/feature_2/data/public/val.parquet}"
  _opt_unl="${UNLABELED_VALIDATOR:-workspace/preprocess/statistical_test/explorer/feature_2/data/validator/validator.parquet}"

  for _p in "${_opt_features}" "${_opt_train}" "${_opt_val}" "${_opt_unl}"; do
    if [[ ! -f "${_p}" ]]; then
      echo "[auto] --run-optuna requires existing file: ${_p}" >&2
      exit 1
    fi
  done

  if [[ -z "${OPTUNA_WORK_DIR}" ]]; then
    _ts="$(date +%Y%m%d_%H%M%S)"
    OPTUNA_WORK_DIR="workspace/ssl_data/ssl_embed/artifacts/optuna_auto_${_ts}"
  fi
  mkdir -p "${OPTUNA_WORK_DIR}"

  _opt_pool="${OPTUNA_WORK_DIR}/ssl_pool.parquet"
  echo "[auto] building optuna SSL pool: ${_opt_pool}"
  PYTHONPATH=. python workspace/ssl_data/ssl_embed/scripts/build_ssl_pool.py \
    --input-parquet "${_opt_train}" \
    --input-parquet "${_opt_unl}" \
    --feature-cols-file "${_opt_features}" \
    --output-parquet "${_opt_pool}" \
    --drop-non-feature-cols

  OPTUNA_CMD=(
    python workspace/ssl_data/ssl_embed/scripts/tune_ssl_lgbm_optuna.py
    --mode ssl_lgbm
    --pool-parquet "${_opt_pool}"
    --feature-cols-file "${_opt_features}"
    --train-parquet "${_opt_train}"
    --val-parquet "${_opt_val}"
    --work-dir "${OPTUNA_WORK_DIR}"
    --outer-trials "${OPTUNA_OUTER_TRIALS}"
    --inner-trials "${OPTUNA_INNER_TRIALS}"
    --objective "${OPTUNA_OBJECTIVE}"
    --fpr-cap "${OPTUNA_FPR_CAP}"
  )
  if [[ -n "${LGBM_DEVICE:-}" ]]; then
    OPTUNA_CMD+=(--lgbm-device "${LGBM_DEVICE}")
  fi
  if [[ ${#TEST_PARQUETS[@]} -gt 0 ]]; then
    for _t in "${TEST_PARQUETS[@]}"; do
      OPTUNA_CMD+=(--holdout-parquet "${_t}")
    done
  fi
  if [[ ${#OPTUNA_EXTRA_ARGS[@]} -gt 0 ]]; then
    OPTUNA_CMD+=("${OPTUNA_EXTRA_ARGS[@]}")
  fi

  echo "[auto] optuna command: ${OPTUNA_CMD[*]}"
  PYTHONPATH=. "${OPTUNA_CMD[@]}"
  MANIFEST_PATH="${OPTUNA_WORK_DIR}/ssl_embed_ablation_manifest.json"
  echo "[auto] optuna manifest: ${MANIFEST_PATH}"
fi

if [[ -z "${MANIFEST_PATH}" ]]; then
  export _AUTO_MANIFEST_SEARCH_DIR="${MANIFEST_SEARCH_DIR}"
  MANIFEST_PATH="$(python3 - <<'PY'
from pathlib import Path
import os

root = Path(os.environ.get("_AUTO_MANIFEST_SEARCH_DIR", "workspace/ssl_data/ssl_embed/artifacts"))
if not root.exists():
    print("")
    raise SystemExit(0)
candidates = sorted(root.rglob("ssl_embed_ablation_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0].as_posix() if candidates else "")
PY
)"
fi

CMD=(bash workspace/ssl_data/ssl_embed/scripts/run_ssl_lgbm_ablation.sh)
if [[ -n "${MANIFEST_PATH}" ]]; then
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Missing manifest: ${MANIFEST_PATH}" >&2
    exit 1
  fi
  echo "[auto] using manifest: ${MANIFEST_PATH}"
  CMD+=(--manifest "${MANIFEST_PATH}")
else
  echo "[auto] no manifest found; running with script defaults"
fi

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

echo "[auto] command: ${CMD[*]}"
if [[ -n "${LGBM_DEVICE:-}" ]]; then
  echo "[auto] LGBM_DEVICE=${LGBM_DEVICE}"
fi
"${CMD[@]}"

