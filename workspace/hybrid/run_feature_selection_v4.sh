#!/usr/bin/env bash
# Feature selection v4: May-8 50% in FIT + unlabeled live Finney coverage (1_0.jsonl).
#
# Prerequisites:
#   - workspace/hybrid/dataset/test/may8_gold_test_features.parquet
#   - workspace/dataset/real_distribution/<log>.jsonl  (unlabeled)
#
# Usage:
#   ./workspace/hybrid/run_feature_selection_v4.sh
#   LIVE_JSONL=workspace/dataset/real_distribution/1_0.jsonl ./workspace/hybrid/run_feature_selection_v4.sh
#   ./workspace/hybrid/run_feature_selection_v4.sh --phase lockbox   # after optuna done
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LIVE_JSONL="${LIVE_JSONL:-workspace/dataset/real_distribution/1_0.jsonl}"
LOG_DIR="workspace/hybrid/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/feature_selection_v4_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[fs-v4] $*" | tee -a "$LOG_FILE"; }

log "=== Feature selection v4 ==="
log "repo: $REPO_ROOT"
log "live jsonl: $LIVE_JSONL"

log "Step 0a: split May-8 → 50% FS train / 50% FS lockbox"
python3 workspace/hybrid/scripts/split_may8_for_feature_selection.py | tee -a "$LOG_FILE"

log "Step 0b: extract live Finney features (50% coverage / 50% monitor, unlabeled)"
if [ -f "$LIVE_JSONL" ]; then
  python3 workspace/hybrid/scripts/extract_live_finney_features.py \
    --jsonl "$LIVE_JSONL" | tee -a "$LOG_FILE"
else
  log "WARN: $LIVE_JSONL not found — skipping live extract (v4 live filter disabled)"
fi

EXTRA=("$@")
if [ ${#EXTRA[@]} -eq 0 ]; then
  EXTRA=(--phase all --fs-v4 --trials 300 --trials-per-x 50 --top 10)
fi

log "Step 1–4: feature_selection_v3.py ${EXTRA[*]}"
PYTHONUNBUFFERED=1 python3 -u workspace/hybrid/scripts/feature_selection_v3.py \
  "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"

log "Done. Log: $LOG_FILE"
log "Winner: workspace/hybrid/selected_features_v3.json"
log "Live profile: workspace/hybrid/bot_system/data/live_profile.json"
