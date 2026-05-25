#!/usr/bin/env bash
# Phase 2 smoke: passive-policy A/B + small match/generate/validate.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
DATA="workspace/hybrid/bot_system/data"
LOG="workspace/hybrid/bot_system/logs"
mkdir -p "$LOG"

echo "=== Phase 2 Step A: A/B Phase-1 knobs vs Phase-2 passive policy ==="
$PY workspace/hybrid/bot_system/22_test_phase2_generator.py 2>&1 | tee "$LOG/phase2_ab.log"

echo ""
echo "=== Phase 2 Step B: match + probe generate (Phase-1 engine + Phase-2 policy) ==="
$PY workspace/hybrid/bot_system/06_build_may8_target.py --hard-only --blend 1.0 --out "$DATA/may8_hard_target_fingerprint.json"
$PY workspace/hybrid/bot_system/03_match_profiles.py \
  --fp "$DATA/may8_hard_target_fingerprint.json" \
  --out "$DATA/may8_phase2_matched.json" \
  --passive --fp-cols may8 --stakes micro \
  --n-candidates 60 --top-k 8 --workers 6 --seed 43

$PY workspace/hybrid/bot_system/04_generate_targeted_bots.py \
  --matched "$DATA/may8_phase2_matched.json" \
  --out "$DATA/may8_phase2_probe.parquet" \
  --passive --source-tag may8_matched_bot \
  --top-k 8 --perturbations-per-seed 2 --chunks-per-profile 15 --workers 4

$PY workspace/hybrid/bot_system/19_validate_may8_generated.py \
  --generated "$DATA/may8_phase2_probe.parquet" \
  --fingerprint "$DATA/may8_hard_target_fingerprint.json"

echo ""
echo "=== Phase 2 smoke done ==="
