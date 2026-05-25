#!/usr/bin/env bash
# Production-style eval: fixed 0.5 vs dynamic threshold (batched requests).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
BUNDLE="${BUNDLE:-workspace/hybrid/model_bundle_may8_reflect}"

$PY workspace/hybrid/bot_system/26_eval_production_dynamic.py \
  --bundle "$BUNDLE" \
  --batch-size 20 \
  --out workspace/hybrid/bot_system/data/production_dynamic_eval.json \
  2>&1 | tee workspace/hybrid/bot_system/logs/production_dynamic_eval.log

echo ""
echo "Enable dynamic threshold in miner:"
echo "  export POKER44_DYNAMIC_THRESHOLD=1"
echo "  export POKER44_MINER_MODEL_BUNDLE_DIR=$BUNDLE"
