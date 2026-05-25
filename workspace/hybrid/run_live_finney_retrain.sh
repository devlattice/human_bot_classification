#!/usr/bin/env bash
# Live Finney retrain: 1_0.jsonl → fingerprint → synthetic bots → v12 train → full eval.
#
# Usage:
#   LIVE_JSONL=workspace/dataset/real_distribution/1_0.jsonl ./workspace/hybrid/run_live_finney_retrain.sh
#   SKIP_GENERATE=1 ./workspace/hybrid/run_live_finney_retrain.sh   # reuse existing bot parquet
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
DATA="workspace/hybrid/bot_system/data"
LOG="workspace/hybrid/bot_system/logs"
BUNDLE="workspace/model/artifacts/model_bundle_v12_prod"
mkdir -p "$DATA" "$LOG" "$(dirname "$BUNDLE")"

LIVE_JSONL="${LIVE_JSONL:-workspace/dataset/real_distribution/1_0.jsonl}"
LIVE_FP="$DATA/live_bot_fingerprint.json"
TARGET_FP="$DATA/live_finney_target_fingerprint.json"
MATCHED="$DATA/live_finney_matched_profiles.json"
GEN_OUT="$DATA/live_finney_bot_features.parquet"

N_CANDIDATES="${N_CANDIDATES:-120}"
TOP_K="${TOP_K:-14}"
PERTURB="${PERTURB:-5}"
CHUNKS_PER_PROFILE="${CHUNKS_PER_PROFILE:-18}"
WORKERS_MATCH="${WORKERS_MATCH:-4}"
WORKERS_GEN="${WORKERS_GEN:-4}"
LIVE_BLEND="${LIVE_BLEND:-0.5}"
MAY8_GOLD="${MAY8_GOLD:-workspace/hybrid/dataset/test/may8_gold_test_features.parquet}"

log() { echo "[live-retrain] $*"; }

log "=== Phase 0: extract live features from JSONL ==="
if [ -f "$LIVE_JSONL" ]; then
  $PY workspace/hybrid/scripts/extract_live_finney_features.py --jsonl "$LIVE_JSONL"
else
  log "WARN: $LIVE_JSONL missing — skip extract"
fi

if [ "${SKIP_GENERATE:-0}" != "1" ]; then
  log "=== Phase 1: live bot fingerprint (02_discover) ==="
  $PY workspace/hybrid/bot_system/02_discover_profile.py \
    --input "$LIVE_JSONL" \
    --fp-out "$LIVE_FP" 2>&1 | tee "$LOG/live_01_discover.log"

  log "=== Phase 2: blended target (May-8 bots + live Finney) ==="
  $PY workspace/hybrid/bot_system/06_build_may8_target.py \
    --gold "$MAY8_GOLD" \
    --date 2026-05-08 \
    --blend "$LIVE_BLEND" \
    --live-fp "$LIVE_FP" \
    --out "$TARGET_FP" 2>&1 | tee "$LOG/live_02_target_fp.log"

  log "=== Phase 3: match profiles ==="
  $PY workspace/hybrid/bot_system/03_match_profiles.py \
    --fp "$TARGET_FP" \
    --out "$MATCHED" \
    --passive \
    --fp-cols may8 \
    --stakes micro \
    --n-candidates "$N_CANDIDATES" \
    --top-k "$TOP_K" \
    --workers "$WORKERS_MATCH" \
    --seed 42 2>&1 | tee "$LOG/live_03_match.log"

  log "=== Phase 4: generate synthetic bots ==="
  $PY workspace/hybrid/bot_system/04_generate_targeted_bots.py \
    --matched "$MATCHED" \
    --out "$GEN_OUT" \
    --passive \
    --source-tag live_finney_matched_bot \
    --top-k "$TOP_K" \
    --perturbations-per-seed "$PERTURB" \
    --chunks-per-profile "$CHUNKS_PER_PROFILE" \
    --workers "$WORKERS_GEN" \
    --per-job-timeout 90 \
    --seed 20260524 2>&1 | tee "$LOG/live_04_generate.log"
else
  log "SKIP_GENERATE=1 — using existing $GEN_OUT"
fi

if [ ! -f "$GEN_OUT" ]; then
  log "ERROR: bot parquet missing: $GEN_OUT"
  exit 1
fi

log "=== Phase 5: train v12 + eval all test sets + real_distribution ==="
$PY workspace/hybrid/scripts/train_v11_may8_blend.py \
  --output-dir "$BUNDLE" \
  --features-json workspace/hybrid/selected_features_v3.json \
  --live-bot-parquet "$GEN_OUT" \
  --live-bot-repeat 2 \
  --live-bot-cap 4000 \
  --real-dist-dir workspace/dataset/real_distribution \
  --eval-json workspace/hybrid/bot_system/data/v12_prod_eval.json \
  2>&1 | tee "$LOG/live_05_train_eval.log"

log "=== Phase 6: patch .env for v12 deploy ==="
BUNDLE_ABS="$(realpath "$BUNDLE")"
"$PY" - <<'PY' "$ROOT/.env" "$BUNDLE_ABS"
import re, sys
from pathlib import Path
env_path, bundle = Path(sys.argv[1]), sys.argv[2]
if not env_path.is_file():
    print("No .env to patch"); raise SystemExit(0)
text = env_path.read_text(encoding="utf-8")
updates = {
    "POKER44_MINER_MODEL_BUNDLE_DIR": bundle,
    "POKER44_MINER_MODEL_PATH": f"{bundle}/model.joblib",
    "POKER44_MINER_TRANSFORM_META_PATH": f"{bundle}/transform_meta.json",
    "POKER44_PRODUCTION_THRESHOLD_JSON": f"{bundle}/production_threshold.json",
}
for k, v in updates.items():
    pat = rf"^{re.escape(k)}=.*$"
    repl = f"{k}={v}"
    if re.search(pat, text, flags=re.M):
        text = re.sub(pat, repl, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n" + repl + "\n"
env_path.write_text(text, encoding="utf-8")
print("Patched .env → v12 bundle")
PY

log "=== Done ==="
log "Bundle: $BUNDLE"
log "Eval:   workspace/hybrid/bot_system/data/v12_prod_eval.json"
log "Deploy: ./workspace/model/deploy_prod.sh --validate-only"
