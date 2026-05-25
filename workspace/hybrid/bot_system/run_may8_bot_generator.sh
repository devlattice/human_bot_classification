#!/usr/bin/env bash
# May-8 reflect bot generator: fingerprint hard May-8 bots → match → generate → train/eval.
#
# Outputs:
#   data/may8_hard_target_fingerprint.json
#   data/may8_reflect_matched_profiles.json
#   data/may8_reflect_bot_features.parquet
#   data/may8_bot_pipeline_results.json
#   ../model_bundle_may8_reflect/
#
# Usage:
#   ./workspace/hybrid/bot_system/run_may8_bot_generator.sh
#   N_CANDIDATES=200 ./workspace/hybrid/bot_system/run_may8_bot_generator.sh
#
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
DATA="workspace/hybrid/bot_system/data"
LOG="workspace/hybrid/bot_system/logs"
mkdir -p "$DATA" "$LOG"

FP="$DATA/may8_hard_target_fingerprint.json"
MATCHED="$DATA/may8_reflect_matched_profiles.json"
GEN_OUT="$DATA/may8_reflect_bot_features.parquet"
RESULTS="$DATA/may8_bot_pipeline_results.json"
BUNDLE="workspace/hybrid/model_bundle_may8_reflect"

N_CANDIDATES="${N_CANDIDATES:-250}"
TOP_K="${TOP_K:-18}"
PERTURB="${PERTURB:-6}"
CHUNKS_PER_PROFILE="${CHUNKS_PER_PROFILE:-25}"
WORKERS_MATCH="${WORKERS_MATCH:-8}"
WORKERS_GEN="${WORKERS_GEN:-$(($(nproc) / 2))}"
PER_JOB_TIMEOUT="${PER_JOB_TIMEOUT:-120}"
MAY8_BOT_CAP="${MAY8_BOT_CAP:-12000}"

echo "============================================================"
echo " May-8 reflect bot generator"
echo " n_candidates=$N_CANDIDATES top_k=$TOP_K perturb=$PERTURB chunks/profile=$CHUNKS_PER_PROFILE"
echo "============================================================"

echo
echo "[1/4] Fingerprint from May-8 HARD bots (hold-out score < 0.5)"
$PY workspace/hybrid/bot_system/06_build_may8_target.py \
    --hard-only \
    --blend 1.0 \
    --out "$FP" 2>&1 | tee "$LOG/may8_reflect_01_fp.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

echo
REFINE_TOP="${REFINE_TOP:-12}"
REFINE_CHUNKS="${REFINE_CHUNKS:-10}"
MATCH_CHUNKS="${MATCH_CHUNKS:-6}"

echo "[2/4] LHS match (passive + May-8-ranked weighted features + micro stakes)"
$PY workspace/hybrid/bot_system/03_match_profiles.py \
    --fp "$FP" \
    --out "$MATCHED" \
    --passive \
    --fp-cols may8 \
    --stakes micro \
    --chunks-per-candidate "$MATCH_CHUNKS" \
    --refine-top "$REFINE_TOP" \
    --refine-chunks "$REFINE_CHUNKS" \
    --n-candidates "$N_CANDIDATES" \
    --top-k "$TOP_K" \
    --workers "$WORKERS_MATCH" \
    --seed 42 2>&1 | tee "$LOG/may8_reflect_02_match.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

echo
echo "[3/4] Generate synthetic May-8-style bot chunks (Phase-1 engine: locked stakes + BB stacks)"
# Phase-1 generator is default in 04_generate_targeted_bots.py; use --legacy-generator to revert.
$PY workspace/hybrid/bot_system/04_generate_targeted_bots.py \
    --matched "$MATCHED" \
    --out "$GEN_OUT" \
    --passive \
    --source-tag may8_matched_bot \
    --top-k "$TOP_K" \
    --perturbations-per-seed "$PERTURB" \
    --chunks-per-profile "$CHUNKS_PER_PROFILE" \
    --workers "$WORKERS_GEN" \
    --per-job-timeout "$PER_JOB_TIMEOUT" \
    --seed 20260508 2>&1 | tee "$LOG/may8_reflect_03_gen.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

echo
echo "[3b] Validate generated features vs May-8 hard gold (KS)"
$PY workspace/hybrid/bot_system/19_validate_may8_generated.py \
    --generated "$GEN_OUT" \
    --fingerprint "$FP" 2>&1 | tee "$LOG/may8_reflect_03b_validate.log"

echo
echo "[4/4] Train (no May-8 gold) + test + May-8 + real_distribution"
$PY workspace/hybrid/bot_system/18_eval_may8_bot_pipeline.py \
    --may8-matched "$GEN_OUT" \
    --may8-bot-cap "$MAY8_BOT_CAP" \
    --bundle-out "$BUNDLE" \
    --results-out "$RESULTS" 2>&1 | tee "$LOG/may8_reflect_04_eval.log"
RC=${PIPESTATUS[0]}

echo
echo "============================================================"
if [ "$RC" = "0" ]; then
    echo " PASS — see $RESULTS"
else
    echo " Done (gates may not pass) — see $RESULTS  exit=$RC"
fi
echo " bundle: $BUNDLE"
echo "============================================================"
exit "$RC"
