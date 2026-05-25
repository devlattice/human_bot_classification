#!/usr/bin/env bash
set -euo pipefail

# SSL embedding + LGBM ablation (research / optional). For production, prefer the tabular
# baseline (e.g. baseline_19 + lgbm_2 on raw features) until SSL clearly matches or beats it on holdouts and OOD.
# Run from repo root:
#   bash workspace/ssl_data/ssl_embed/scripts/run_ssl_lgbm_ablation.sh
# Optional CLI (override env if both are set; CLI wins):
#   bash .../run_ssl_lgbm_ablation.sh \
#     --input-parquet workspace/preprocess/statistical_test/explorer/feature_2/data/public/train.parquet \
#     --validator-parquet workspace/preprocess/statistical_test/explorer/feature_2/data/validator/validator.parquet
#
# After tune_ssl_lgbm_optuna.py, pass the manifest from the same --work-dir you used:
#   --manifest workspace/.../optuna_nested/ssl_embed_ablation_manifest.json   # nested SSL×LGBM
#   --manifest workspace/.../optuna_mask_only/ssl_embed_ablation_manifest.json  # --freeze-lgbm only

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

SSL_EMBED_ABLATION_MANIFEST="${SSL_EMBED_ABLATION_MANIFEST:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-parquet)
      SRC_TRAIN="${2:?--input-parquet requires a path}"
      shift 2
      ;;
    --validator-parquet)
      UNLABELED_VALIDATOR="${2:?--validator-parquet requires a path}"
      shift 2
      ;;
    --manifest)
      SSL_EMBED_ABLATION_MANIFEST="${2:?--manifest requires a path}"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE' >&2
Usage: bash workspace/ssl_data/ssl_embed/scripts/run_ssl_lgbm_ablation.sh [options]

Options:
  --input-parquet PATH      Labeled source train parquet (sets SRC_TRAIN)
  --validator-parquet PATH  Unlabeled validator parquet for SSL pool (sets UNLABELED_VALIDATOR)
  --manifest PATH           ssl_embed_ablation_manifest.json from tune_ssl_lgbm_optuna.py (ssl_lgbm mode)
  -h, --help                Show this help

Environment variables still apply (e.g. SRC_TRAIN, UNLABELED_VALIDATOR, TEST_PUB1..5, RUN_ROOT).
CLI flags set input parquets, validator, and optional manifest when passed.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

# Defaults for SSL pretrain + LGBM (overridden by ssl_embed_ablation_manifest via shell-exports).
SSL_MASK_RATIO="${SSL_MASK_RATIO:-0.30}"
SSL_MASK_MODE="${SSL_MASK_MODE:-random}"
SSL_MASK_MIXED_ALPHA="${SSL_MASK_MIXED_ALPHA:-0.30}"
SSL_EMBED_DIM="${SSL_EMBED_DIM:-32}"
SSL_HIDDEN_DIM="${SSL_HIDDEN_DIM:-96}"
SSL_MAX_ITER="${SSL_MAX_ITER:-80}"
SSL_SEED="${SSL_SEED:-42}"
SSL_MASK_WEIGHT_JSON="${SSL_MASK_WEIGHT_JSON:-}"
LGBM_DEVICE="${LGBM_DEVICE:-gpu}"
# Optional DBF-aware defaults. Set USE_DBF_INPUTS=1 to switch default parquet/config
# paths to *_with_dbf variants while still allowing CLI/env overrides.
USE_DBF_INPUTS="${USE_DBF_INPUTS:-0}"

if [[ -n "${SSL_EMBED_ABLATION_MANIFEST}" ]]; then
  if [[ ! -f "${SSL_EMBED_ABLATION_MANIFEST}" ]]; then
    echo "Missing --manifest file: ${SSL_EMBED_ABLATION_MANIFEST}" >&2
    echo "Hint: the manifest path must be exactly tune_ssl_lgbm_optuna.py --work-dir + /ssl_embed_ablation_manifest.json" >&2
    echo "      (e.g. .../optuna_mask_only/ if you used --work-dir .../optuna_mask_only, not necessarily .../optuna_nested/)." >&2
    _mf="$(dirname "${SSL_EMBED_ABLATION_MANIFEST}")"
    _run="$(dirname "${_mf}")"
    if [[ -d "${_run}" ]]; then
      _found="$(find "${_run}" -maxdepth 3 -name ssl_embed_ablation_manifest.json -print 2>/dev/null | head -5)"
      if [[ -n "${_found}" ]]; then
        echo "Found under the same run directory:" >&2
        echo "${_found}" >&2
      fi
    fi
    exit 1
  fi
  echo "[config] Applying ssl_embed_ablation_manifest: ${SSL_EMBED_ABLATION_MANIFEST}"
  eval "$(python3 workspace/ssl_data/ssl_embed/scripts/ssl_embed_ablation_manifest.py shell-exports --manifest "${SSL_EMBED_ABLATION_MANIFEST}")"
fi

TEST_DIR="${TEST_DIR:-workspace/preprocess/statistical_test/explorer/feature_2/data/test}"
if [[ "${USE_DBF_INPUTS}" == "1" ]]; then
  DEFAULT_FEATURES_TXT="workspace/preprocess/statistical_test/explorer/feature_2/config/keep_features_with_dbf.txt"
  DEFAULT_SRC_TRAIN="workspace/preprocess/statistical_test/explorer/feature_2/data/public/train_with_dbf.parquet"
  # Optional: if this does not exist, script will auto-split SRC_TRAIN into train/val.
  DEFAULT_SRC_VAL="workspace/preprocess/statistical_test/explorer/feature_2/data/public/val_with_dbf.parquet"
  # Unlabeled validator-like parquet used for SSL pool.
  DEFAULT_UNLABELED_VALIDATOR="workspace/preprocess/statistical_test/explorer/feature_2/data/validator/validator_with_dbf.parquet"
  DEFAULT_TEST_PUB1="${TEST_DIR}/pb_1_with_dbf.parquet"
  DEFAULT_TEST_PUB2="${TEST_DIR}/pb_2_with_dbf.parquet"
  DEFAULT_TEST_PUB3="${TEST_DIR}/holdout_1_with_dbf.parquet"
  DEFAULT_TEST_PUB4="${TEST_DIR}/holdout_2_with_dbf.parquet"
  DEFAULT_TEST_PUB5="workspace/preprocess/statistical_test/explorer/feature_2/data/irc/irc_val_with_dbf.parquet"
else
  DEFAULT_FEATURES_TXT="workspace/preprocess/statistical_test/explorer/feature_2/config/keep_features.txt"
  DEFAULT_SRC_TRAIN="workspace/ssl_data/SSL/data/ssl_47_mcp055/train.parquet"
  # Optional: if this does not exist, script will auto-split SRC_TRAIN into train/val.
  DEFAULT_SRC_VAL="workspace/ssl_data/SSL/data/ssl_47_mcp055/val.parquet"
  # Unlabeled validator-like parquet used for SSL pool.
  DEFAULT_UNLABELED_VALIDATOR="workspace/preprocess/statistical_test/explorer/feature_2/data/validator/validator.parquet"
  DEFAULT_TEST_PUB1="${TEST_DIR}/pb_1.parquet"
  DEFAULT_TEST_PUB2="${TEST_DIR}/pb_2.parquet"
  DEFAULT_TEST_PUB3="${TEST_DIR}/holdout_1.parquet"
  DEFAULT_TEST_PUB4="${TEST_DIR}/holdout_2.parquet"
  # No irc_test.parquet in-tree; use irc_val for the fifth eval slice (override with TEST_PUB5=.../irc_train.parquet if needed).
  DEFAULT_TEST_PUB5="workspace/preprocess/statistical_test/explorer/feature_2/data/irc/irc_val.parquet"
fi

FEATURES_TXT="${FEATURES_TXT:-${DEFAULT_FEATURES_TXT}}"
SRC_TRAIN="${SRC_TRAIN:-${DEFAULT_SRC_TRAIN}}"
SRC_VAL="${SRC_VAL:-${DEFAULT_SRC_VAL}}"
UNLABELED_VALIDATOR="${UNLABELED_VALIDATOR:-${DEFAULT_UNLABELED_VALIDATOR}}"
TEST_PUB1="${TEST_PUB1:-${DEFAULT_TEST_PUB1}}"
TEST_PUB2="${TEST_PUB2:-${DEFAULT_TEST_PUB2}}"
# Alias support requested for automation API.
TEST_PUB3="${TEST_PUB3:-${DEFAULT_TEST_PUB3}}"
TEST_PUB4="${TEST_PUB4:-${DEFAULT_TEST_PUB4}}"
TEST_PUB5="${TEST_PUB5:-${DEFAULT_TEST_PUB5}}"
TEST_H1="${TEST_H1:-${TEST_PUB3}}"
TEST_H2="${TEST_H2:-${TEST_PUB4}}"
TEST_IRC="${TEST_IRC:-${TEST_PUB5}}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  ARTIFACT_BASE="workspace/ssl_data/ssl_embed/artifacts"
  mkdir -p "${ARTIFACT_BASE}"
  NEXT_VER="$(python3 - <<'PY'
from pathlib import Path
import re
base = Path("workspace/ssl_data/ssl_embed/artifacts")
pat = re.compile(r"^ssl_embed_v(\d+)$")
mx = 0
for p in base.iterdir():
    if p.is_dir():
        m = pat.match(p.name)
        if m:
            mx = max(mx, int(m.group(1)))
print(mx + 1)
PY
)"
  RUN_ROOT="${ARTIFACT_BASE}/ssl_embed_v${NEXT_VER}"
fi

POOL_PQ="${RUN_ROOT}/ssl_pool.parquet"
SSL_DIR="${RUN_ROOT}/ssl_model"
# Single pipeline: embeddings parquet always includes emb_* + original features (see export_embeddings.py default).
EMB_DIR="${RUN_ROOT}/embeddings"
LGBM_DATA_DIR="${RUN_ROOT}/lgbm_data"
LGBM_OUT="${RUN_ROOT}/lgbm_out"
CROSS_OUT="${RUN_ROOT}/cross_eval"

mkdir -p "${RUN_ROOT}" "${LGBM_DATA_DIR}"

if [[ ! -f "${FEATURES_TXT}" ]]; then
  echo "Missing FEATURES_TXT: ${FEATURES_TXT}" >&2
  exit 1
fi
if [[ ! -f "${SRC_TRAIN}" ]]; then
  echo "Missing SRC_TRAIN: ${SRC_TRAIN}" >&2
  exit 1
fi
if [[ ! -f "${UNLABELED_VALIDATOR}" ]]; then
  echo "Missing UNLABELED_VALIDATOR: ${UNLABELED_VALIDATOR}" >&2
  exit 1
fi

for _pq in "${TEST_PUB1}" "${TEST_PUB2}" "${TEST_H1}" "${TEST_H2}" "${TEST_IRC}"; do
  if [[ ! -f "${_pq}" ]]; then
    echo "Missing test parquet: ${_pq}" >&2
    echo "Set TEST_PUB1..TEST_PUB5 (or TEST_H1/TEST_H2/TEST_IRC), or TEST_DIR. Contents of ${TEST_DIR}:" >&2
    ls -la "${TEST_DIR}" 2>/dev/null || true
    exit 1
  fi
done

# export_embeddings names outputs from input basenames; cross-eval must use the same files.
TEST_PUB1_BASE="$(basename "${TEST_PUB1}")"
TEST_PUB2_BASE="$(basename "${TEST_PUB2}")"
TEST_H1_BASE="$(basename "${TEST_H1}")"
TEST_H2_BASE="$(basename "${TEST_H2}")"
TEST_IRC_BASE="$(basename "${TEST_IRC}")"

echo "[config] RUN_ROOT=${RUN_ROOT}"
echo "[config] USE_DBF_INPUTS=${USE_DBF_INPUTS}"
echo "[config] SRC_TRAIN=${SRC_TRAIN}"
echo "[config] UNLABELED_VALIDATOR=${UNLABELED_VALIDATOR}"
echo "[config] TEST_PUB1=${TEST_PUB1}"
echo "[config] TEST_PUB2=${TEST_PUB2}"
echo "[config] TEST_PUB3=${TEST_H1}"
echo "[config] TEST_PUB4=${TEST_H2}"
echo "[config] TEST_PUB5=${TEST_IRC}"
if [[ -n "${SSL_EMBED_ABLATION_MANIFEST:-}" ]]; then
  echo "[config] SSL_MASK_MODE=${SSL_MASK_MODE} SSL_EMBED_DIM=${SSL_EMBED_DIM} LGBM_DEVICE=${LGBM_DEVICE}"
fi

if [[ ! -f "${SRC_VAL}" ]]; then
  echo "[prep] SRC_VAL not found. Creating deterministic 90/10 split from SRC_TRAIN." >&2
  python - <<PY
from pathlib import Path
import pandas as pd

train_path = Path("${SRC_TRAIN}")
val_path = Path("${SRC_VAL}")
tmp_train_path = Path("${RUN_ROOT}") / "split_train.parquet"

df = pd.read_parquet(train_path)
if "label" not in df.columns:
    raise SystemExit(f"Expected 'label' column in {train_path}")
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_val = max(1, int(round(len(df) * 0.10)))
val_df = df.iloc[:n_val].copy()
train_df = df.iloc[n_val:].copy()

tmp_train_path.parent.mkdir(parents=True, exist_ok=True)
train_df.to_parquet(tmp_train_path, index=False)
val_path.parent.mkdir(parents=True, exist_ok=True)
val_df.to_parquet(val_path, index=False)

print(f"[prep] wrote split train: {tmp_train_path} rows={len(train_df)}")
print(f"[prep] wrote split val:   {val_path} rows={len(val_df)}")
PY
  SRC_TRAIN="${RUN_ROOT}/split_train.parquet"
fi

echo "[1/5] Build pooled SSL parquet"
PYTHONPATH=. python workspace/ssl_data/ssl_embed/scripts/build_ssl_pool.py \
  --input-parquet "${SRC_TRAIN}" \
  --input-parquet "${UNLABELED_VALIDATOR}" \
  --feature-cols-file "${FEATURES_TXT}" \
  --output-parquet "${POOL_PQ}" \
  --drop-non-feature-cols

echo "[2/5] Pretrain masked AE"
PRETRAIN_EXTRA=()
if [[ -n "${SSL_MASK_WEIGHT_JSON:-}" ]] && [[ "${SSL_MASK_MODE}" == "mixed" || "${SSL_MASK_MODE}" == "weighted" ]]; then
  PRETRAIN_EXTRA+=(--mask-weight-json "${SSL_MASK_WEIGHT_JSON}")
fi
PYTHONPATH=. python workspace/ssl_data/ssl_embed/scripts/pretrain_masked_ae.py \
  --pool-parquet "${POOL_PQ}" \
  --feature-cols-file "${FEATURES_TXT}" \
  --out-dir "${SSL_DIR}" \
  --mask-ratio "${SSL_MASK_RATIO}" \
  --mask-mode "${SSL_MASK_MODE}" \
  --mask-mixed-alpha "${SSL_MASK_MIXED_ALPHA}" \
  --embed-dim "${SSL_EMBED_DIM}" \
  --hidden-dim "${SSL_HIDDEN_DIM}" \
  --max-iter "${SSL_MAX_ITER}" \
  --seed "${SSL_SEED}" \
  "${PRETRAIN_EXTRA[@]}"

echo "[3/5] Export embeddings + original features (default in export_embeddings.py)"
# Include unlabeled validator so embeddings/ holds validator.parquet (same basename as input) for drift / unsupervised checks.
PYTHONPATH=. python workspace/ssl_data/ssl_embed/scripts/export_embeddings.py \
  --artifact "${SSL_DIR}/ssl_masked_ae.npz" \
  --in-parquet "${SRC_TRAIN}" \
  --in-parquet "${SRC_VAL}" \
  --in-parquet "${UNLABELED_VALIDATOR}" \
  --in-parquet "${TEST_PUB1}" \
  --in-parquet "${TEST_PUB2}" \
  --in-parquet "${TEST_H1}" \
  --in-parquet "${TEST_H2}" \
  --in-parquet "${TEST_IRC}" \
  --out-dir "${EMB_DIR}"

SRC_TRAIN_BASE="$(basename "${SRC_TRAIN}")"
SRC_VAL_BASE="$(basename "${SRC_VAL}")"
cp -f "${EMB_DIR}/${SRC_TRAIN_BASE}" "${LGBM_DATA_DIR}/train.parquet"
cp -f "${EMB_DIR}/${SRC_VAL_BASE}" "${LGBM_DATA_DIR}/val.parquet"

echo "[4/5] Train LGBM on embeddings + original"
LGBM_EXTRA=()
if [[ -n "${SSL_EMBED_ABLATION_MANIFEST:-}" ]]; then
  LGBM_EXTRA+=(--hparams-json "${SSL_EMBED_ABLATION_MANIFEST}" --device "${LGBM_DEVICE}" --log-every 0)
fi
PYTHONPATH=. python workspace/model/scripts/lgbm_2.py \
  --data-dir "${LGBM_DATA_DIR}" \
  --out-dir "${LGBM_OUT}" \
  --seed 42 \
  "${LGBM_EXTRA[@]}"

echo "[5/5] Cross-eval on test parquets"
PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
  --model "${LGBM_OUT}/lgbm_b_classifier.joblib" \
  --eval-parquet "${EMB_DIR}/${TEST_PUB1_BASE}" \
  --eval-parquet "${EMB_DIR}/${TEST_PUB2_BASE}" \
  --eval-parquet "${EMB_DIR}/${TEST_H1_BASE}" \
  --eval-parquet "${EMB_DIR}/${TEST_H2_BASE}" \
  --eval-parquet "${EMB_DIR}/${TEST_IRC_BASE}" \
  --out-dir "${CROSS_OUT}" \
  --threshold 0.5

echo "Done. Outputs under ${RUN_ROOT}"

