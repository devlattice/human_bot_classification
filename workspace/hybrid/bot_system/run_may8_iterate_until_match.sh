#!/usr/bin/env bash
# Repeat match + probe-generate until KS vs May-8 gold bot profile is good enough,
# then full generate + train/eval.
#
# Usage:
#   ./workspace/hybrid/bot_system/run_may8_iterate_until_match.sh
#   MAX_ROUNDS=12 MAX_MEDIAN_KS=0.40 ./workspace/hybrid/bot_system/run_may8_iterate_until_match.sh
#
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export MAX_ROUNDS="${MAX_ROUNDS:-8}"
export MAX_MEDIAN_KS="${MAX_MEDIAN_KS:-0.45}"
export MAX_WEIGHTED_MEDIAN_KS="${MAX_WEIGHTED_MEDIAN_KS:-0.50}"
export N_CANDIDATES_START="${N_CANDIDATES_START:-80}"
export N_CANDIDATES_STEP="${N_CANDIDATES_STEP:-40}"

exec python3 workspace/hybrid/bot_system/20_iterate_until_may8_profile.py \
  --max-rounds "$MAX_ROUNDS" \
  --max-median-ks "$MAX_MEDIAN_KS" \
  --max-weighted-median-ks "$MAX_WEIGHTED_MEDIAN_KS" \
  --n-candidates-start "$N_CANDIDATES_START" \
  --n-candidates-step "$N_CANDIDATES_STEP" \
  "$@"
