#!/usr/bin/env bash
# Inverse-fingerprint loop: progressively tune the bot generator to reproduce
# May-8 gold bot behaviour, then train w/o May-8 and verify recall on it.
#
# Each round:
#   1. Build/refresh the May-8 target fingerprint (optional blend with live).
#   2. LHS sweep over BotProfile knobs + game config; keep top-K matches.
#   3. Bulk-generate chunks from top profiles (with perturbations).
#   4. Train RandomForest WITHOUT May-8 gold, evaluate on held-out May-8 /
#      May-7 / zenodo-test / public-test / acpc-bot-test.
#   5. If pass criteria met → stop. Otherwise widen ranges and try again.
#
# Time cost: hours per round. Designed to be re-runnable; outputs are stamped
# with round index.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
LOG_DIR="workspace/hybrid/bot_system/logs"
mkdir -p "$LOG_DIR"

MAX_ROUNDS="${MAX_ROUNDS:-5}"
BLEND="${BLEND:-0.7}"            # May-8 weight in fingerprint (1.0 = pure)
N_CANDIDATES_START="${N_CANDIDATES_START:-300}"
TOPK_START="${TOPK_START:-20}"
PERTURB_START="${PERTURB_START:-8}"
CHUNKS_PER_PROFILE_START="${CHUNKS_PER_PROFILE_START:-40}"
WORKERS_MATCH="${WORKERS_MATCH:-$(nproc)}"
WORKERS_GEN="${WORKERS_GEN:-$(($(nproc) / 2))}"
PER_JOB_TIMEOUT="${PER_JOB_TIMEOUT:-120}"

# Pass criteria
MIN_MAY8_RECALL="${MIN_MAY8_RECALL:-80}"
MAX_HUMAN_FPR="${MAX_HUMAN_FPR:-2}"
MAX_ZENODO_FPR="${MAX_ZENODO_FPR:-2}"

DATA_DIR="workspace/hybrid/bot_system/data"
FP_PATH="$DATA_DIR/may8_target_fingerprint.json"

echo "============================================================"
echo " Inverse fingerprint loop (max rounds=$MAX_ROUNDS)"
echo " blend may8/live = $BLEND"
echo " pass: may8_recall>=$MIN_MAY8_RECALL  human_fpr<=$MAX_HUMAN_FPR  zenodo_fpr<=$MAX_ZENODO_FPR"
echo "============================================================"

# Step 0: build May-8 target once (it doesn't depend on the round).
echo
echo "[step 0] building May-8 target fingerprint (blend=$BLEND)"
$PY workspace/hybrid/bot_system/06_build_may8_target.py \
    --blend "$BLEND" \
    --out "$FP_PATH" 2>&1 | tee "$LOG_DIR/00_fingerprint.log"

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
    # Widen the search a little each round if we have not passed yet.
    N_CAND=$((N_CANDIDATES_START * ROUND))
    TOPK=$((TOPK_START + (ROUND - 1) * 5))
    PERTURB=$((PERTURB_START + (ROUND - 1) * 4))
    CHUNKS=$((CHUNKS_PER_PROFILE_START + (ROUND - 1) * 20))

    MATCHED_OUT="$DATA_DIR/may8_matched_profiles_r${ROUND}.json"
    GEN_OUT="$DATA_DIR/may8_matched_bot_features.parquet"

    echo
    echo "============================================================"
    echo " ROUND $ROUND  n_candidates=$N_CAND  top_k=$TOPK  perturb=$PERTURB  chunks/profile=$CHUNKS"
    echo "============================================================"

    echo "[step 1] LHS match → $MATCHED_OUT"
    $PY workspace/hybrid/bot_system/03_match_profiles.py \
        --fp "$FP_PATH" \
        --out "$MATCHED_OUT" \
        --n-candidates "$N_CAND" \
        --top-k "$TOPK" \
        --workers "$WORKERS_MATCH" \
        --seed "$((42 + ROUND))" 2>&1 | tee "$LOG_DIR/r${ROUND}_01_match.log"
    if [ "${PIPESTATUS[0]}" != "0" ]; then
        echo "[abort] match step failed at round $ROUND"
        break
    fi

    echo "[step 2] generate → $GEN_OUT"
    $PY workspace/hybrid/bot_system/04_generate_targeted_bots.py \
        --matched "$MATCHED_OUT" \
        --out "$GEN_OUT" \
        --top-k "$TOPK" \
        --perturbations-per-seed "$PERTURB" \
        --chunks-per-profile "$CHUNKS" \
        --workers "$WORKERS_GEN" \
        --per-job-timeout "$PER_JOB_TIMEOUT" \
        --seed "$((2026 + ROUND))" 2>&1 | tee "$LOG_DIR/r${ROUND}_02_gen.log"
    if [ "${PIPESTATUS[0]}" != "0" ]; then
        echo "[abort] gen step failed at round $ROUND"
        break
    fi

    echo "[step 3] train w/o May-8 and check"
    $PY workspace/hybrid/bot_system/09_train_and_check.py \
        --round "$ROUND" \
        --may8-matched "$GEN_OUT" \
        --min-may8-recall "$MIN_MAY8_RECALL" \
        --max-human-fpr "$MAX_HUMAN_FPR" \
        --max-zenodo-fpr "$MAX_ZENODO_FPR" 2>&1 | tee "$LOG_DIR/r${ROUND}_03_train.log"
    RC=${PIPESTATUS[0]}

    if [ "$RC" = "0" ]; then
        echo
        echo "============================================================"
        echo " PASS at round $ROUND"
        echo " bundle: $DATA_DIR/round_$(printf '%02d' "$ROUND")_bundle/"
        echo "============================================================"
        exit 0
    elif [ "$RC" = "10" ]; then
        echo "[round $ROUND] soft-fail; widening and retrying"
        continue
    else
        echo "[abort] train/eval failed with rc=$RC"
        break
    fi
done

echo
echo "============================================================"
echo " Did not pass after $MAX_ROUNDS rounds. Check $LOG_DIR/ for diagnostics."
echo "============================================================"
exit 1
