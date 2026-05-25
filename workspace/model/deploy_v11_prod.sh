#!/usr/bin/env bash
# Train (or refresh) v11 production bundle under workspace/model/artifacts/ and print deploy env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUNDLE_REL="workspace/model/artifacts/model_bundle_v11_prod"
BUNDLE_DIR="$REPO_ROOT/$BUNDLE_REL"
ENV_FILE="${MINER_ENV_FILE:-$REPO_ROOT/.env}"

log() { echo "[deploy-v11] $*"; }

if [ "${1:-}" = "--validate-only" ]; then
  log "Validate-only mode (no retrain)"
  python3 workspace/model/scripts/validate_deployment.py
  exit $?
fi

log "Training v11 prod bundle -> $BUNDLE_REL"
python3 workspace/hybrid/scripts/train_v11_may8_blend.py --output-dir "$BUNDLE_DIR"

required=(
  model.joblib
  feature_cols.json
  transform_meta.json
  production_threshold.json
  retrain_summary.json
)
for f in "${required[@]}"; do
  if [ ! -f "$BUNDLE_DIR/$f" ]; then
    log "ERROR: missing $BUNDLE_DIR/$f"
    exit 1
  fi
done

MODEL_PATH="$(realpath "$BUNDLE_DIR/model.joblib")"
TRANSFORM_PATH="$(realpath "$BUNDLE_DIR/transform_meta.json")"
THRESHOLD_PATH="$(realpath "$BUNDLE_DIR/production_threshold.json")"
BUNDLE_ABS="$(realpath "$BUNDLE_DIR")"

log ""
log "═══════════════════════════════════════════════════════════════"
log "DEPLOYMENT (canonical: workspace/model/artifacts/)"
log "═══════════════════════════════════════════════════════════════"
log "Set in .env (or export before run_miner.sh):"
log "  POKER44_MINER_MODEL_BUNDLE_DIR=$BUNDLE_ABS"
log "  POKER44_MINER_MODEL_PATH=$MODEL_PATH"
log "  POKER44_MINER_TRANSFORM_META_PATH=$TRANSFORM_PATH"
log "  POKER44_PRODUCTION_THRESHOLD_JSON=$THRESHOLD_PATH"
log "  POKER44_DYNAMIC_THRESHOLD=0"
log "  POKER44_MINER_OTHER_ONLY=0"
log "  POKER44_MINER_REQUIRE_MODEL=1"
log ""
log "Restart miner:"
log "  ./scripts/miner/run/run_miner.sh"
log ""

if [ -f "$ENV_FILE" ]; then
  log "Patching $ENV_FILE model paths..."
  python3 - <<'PY' "$ENV_FILE" "$BUNDLE_ABS" "$MODEL_PATH" "$TRANSFORM_PATH" "$THRESHOLD_PATH"
import re
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
bundle = sys.argv[2]
model = sys.argv[3]
transform = sys.argv[4]
threshold = sys.argv[5]

updates = {
    "POKER44_MINER_MODEL_BUNDLE_DIR": bundle,
    "POKER44_MINER_MODEL_PATH": model,
    "POKER44_MINER_TRANSFORM_META_PATH": transform,
    "POKER44_PRODUCTION_THRESHOLD_JSON": threshold,
}
text = env_path.read_text(encoding="utf-8")
for key, val in updates.items():
    pat = rf"^{re.escape(key)}=.*$"
    repl = f"{key}={val}"
    if re.search(pat, text, flags=re.M):
        text = re.sub(pat, repl, text, count=1, flags=re.M)
    else:
        text = text.rstrip() + "\n" + repl + "\n"
env_path.write_text(text, encoding="utf-8")
print(f"Updated {env_path}")
PY
else
  log "No .env at $ENV_FILE — set paths manually."
fi

log "Running deployment validation..."
if python3 workspace/model/scripts/validate_deployment.py; then
  log "Validation: PASS"
else
  log "Validation: FAIL"
  exit 1
fi

log "Done."
