#!/usr/bin/env bash
# Deploy production miner bundle (default: model_bundle_v12_prod).
#
# Usage:
#   ./workspace/model/deploy_prod.sh --validate-only
#   BUNDLE_REL=workspace/model/artifacts/model_bundle_v12_prod ./workspace/model/deploy_prod.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUNDLE_REL="${BUNDLE_REL:-workspace/model/artifacts/model_bundle_v12_prod}"
BUNDLE_DIR="$REPO_ROOT/$BUNDLE_REL"
ENV_FILE="${MINER_ENV_FILE:-$REPO_ROOT/.env}"

log() { echo "[deploy] $*"; }

if [ "${1:-}" = "--validate-only" ]; then
  log "Validate-only: $BUNDLE_REL"
  BUNDLE_DIR="$BUNDLE_DIR" python3 workspace/model/scripts/validate_deployment.py
  exit $?
fi

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
    log "Train first: ./workspace/hybrid/run_live_finney_retrain.sh"
    exit 1
  fi
done

MODEL_PATH="$(realpath "$BUNDLE_DIR/model.joblib")"
TRANSFORM_PATH="$(realpath "$BUNDLE_DIR/transform_meta.json")"
THRESHOLD_PATH="$(realpath "$BUNDLE_DIR/production_threshold.json")"
BUNDLE_ABS="$(realpath "$BUNDLE_DIR")"
THR="$(python3 -c "import json; print(json.load(open('$THRESHOLD_PATH'))['selected_threshold'])")"

log ""
log "═══════════════════════════════════════════════════════════════"
log "PRODUCTION DEPLOY: $BUNDLE_REL"
log "  threshold=$THR  (POKER44_DYNAMIC_THRESHOLD=0)"
log "═══════════════════════════════════════════════════════════════"

if [ -f "$ENV_FILE" ]; then
  log "Patching $ENV_FILE ..."
  python3 - <<'PY' "$ENV_FILE" "$BUNDLE_ABS" "$MODEL_PATH" "$TRANSFORM_PATH" "$THRESHOLD_PATH" "$THR"
import json
import re
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
bundle, model, transform, threshold_path = sys.argv[2:6]
thr = sys.argv[6]

updates = {
    "POKER44_MINER_MODEL_BUNDLE_DIR": bundle,
    "POKER44_MINER_MODEL_PATH": model,
    "POKER44_MINER_TRANSFORM_META_PATH": transform,
    "POKER44_PRODUCTION_THRESHOLD_JSON": threshold_path,
    "POKER44_THRESHOLD_FALLBACK": thr,
    "POKER44_DYNAMIC_THRESHOLD": "0",
    "POKER44_MINER_OTHER_ONLY": "0",
    "POKER44_MINER_REQUIRE_MODEL": "1",
    "POKER44_MODEL_NAME": "hybrid-rf-v12-live-finney",
    "POKER44_MODEL_VERSION": "12.0.0",
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
  log "No .env at $ENV_FILE — set paths manually (see DEPLOY.md)."
fi

log "Running deployment validation..."
python3 workspace/model/scripts/validate_deployment.py

log ""
log "Restart miner: ./scripts/miner/run/run_miner.sh"
log "Expect log: using fixed threshold=$THR"
log "Done."
