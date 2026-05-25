#!/usr/bin/env bash
# Hard-bot-focused retrain (Phase-3 synthetic) + test / May-8 / real_distribution eval.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
DATA="workspace/hybrid/bot_system/data"
LOG="workspace/hybrid/bot_system/logs"
mkdir -p "$LOG"

SYN="${SYN:-$DATA/may8_reflect_bot_features.parquet}"
REPEAT="${MAY8_REPEAT:-5}"
WEIGHT="${MAY8_ROW_WEIGHT:-2.5}"

echo "=== Hard-focus retrain + eval ==="
$PY workspace/hybrid/bot_system/24_retrain_may8_hard_focus.py \
  --may8-synthetic "$SYN" \
  --may8-repeat "$REPEAT" \
  --may8-row-weight "$WEIGHT" \
  --may8-cap 8000 \
  --bundle-out workspace/hybrid/model_bundle_may8_hard_focus \
  --results-out "$DATA/may8_hard_focus_eval_results.json" \
  2>&1 | tee "$LOG/may8_hard_focus_eval.log"

echo ""
echo "=== real_distribution detail (CSV) ==="
$PY workspace/hybrid/bot_system/15_score_real_distribution.py \
  --bundle workspace/hybrid/model_bundle_may8_hard_focus \
  --out-csv "$DATA/real_distribution_hard_focus_scores.csv" \
  --out-summary "$DATA/real_distribution_hard_focus_summary.txt"

echo ""
echo "=== done ==="
