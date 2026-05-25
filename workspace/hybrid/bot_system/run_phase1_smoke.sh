#!/usr/bin/env bash
# Phase 1 smoke: A/B KS test + small May-8 match/generate with fixed engine.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"
DATA="workspace/hybrid/bot_system/data"
LOG="workspace/hybrid/bot_system/logs"
mkdir -p "$LOG"

echo "=== Phase 1 Step A: A/B legacy vs fixed generator ==="
$PY workspace/hybrid/bot_system/21_test_phase1_generator.py 2>&1 | tee "$LOG/phase1_ab.log"

echo ""
echo "=== Phase 1 Step B: match + probe generate (fixed engine) ==="
$PY workspace/hybrid/bot_system/06_build_may8_target.py --hard-only --blend 1.0 --out "$DATA/may8_hard_target_fingerprint.json"
$PY workspace/hybrid/bot_system/03_match_profiles.py \
  --fp "$DATA/may8_hard_target_fingerprint.json" \
  --out "$DATA/may8_phase1_matched.json" \
  --passive --fp-cols may8 --stakes micro \
  --n-candidates 60 --top-k 8 --workers 6 --seed 42

$PY workspace/hybrid/bot_system/04_generate_targeted_bots.py \
  --matched "$DATA/may8_phase1_matched.json" \
  --out "$DATA/may8_phase1_probe.parquet" \
  --passive --source-tag may8_matched_bot \
  --top-k 8 --perturbations-per-seed 2 --chunks-per-profile 15 --workers 4

$PY workspace/hybrid/bot_system/19_validate_may8_generated.py \
  --generated "$DATA/may8_phase1_probe.parquet" \
  --fingerprint "$DATA/may8_hard_target_fingerprint.json"

echo ""
echo "=== Phase 1 smoke done ==="
