#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Poker44 Miner-B Daily Retrain Pipeline
# ═══════════════════════════════════════════════════════════════════
#
# This script automates the full retrain cycle:
#   1. Download latest gold data from benchmark API
#   2. Extract features from gold data
#   3. (Optional) Generate full-spectrum bot profiles if not present
#   4. Train production model with mixed data
#   5. Deploy model to miner
#
# Usage:
#   ./workspace/hybrid/retrain.sh              # full pipeline
#   ./workspace/hybrid/retrain.sh --skip-bots  # skip bot generation
#   ./workspace/hybrid/retrain.sh --skip-download  # skip gold download
#
# Cron example (daily at 2 AM UTC):
#   0 2 * * * cd /home/dr/Workspace/Poker44-subnet && ./workspace/hybrid/retrain.sh >> workspace/hybrid/retrain.log 2>&1
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GOLD_DIR="workspace/dataset/source/gold_dataset"
GOLD_FEATURES="workspace/hybrid/dataset/gold_features.parquet"
ZENODO_FEATURES="workspace/hybrid/dataset/zenodo_features.parquet"
PUBLIC_FEATURES="workspace/hybrid/dataset/public_features.parquet"
FULL_SPECTRUM="workspace/hybrid/full_spectrum_bot_features.parquet"
MODEL_BUNDLE="workspace/hybrid/model_bundle"
ENV_FILE=".env"

SKIP_BOTS=false
SKIP_DOWNLOAD=false
SKIP_EXTRACT_STATIC=false

for arg in "$@"; do
    case "$arg" in
        --skip-bots) SKIP_BOTS=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-extract-static) SKIP_EXTRACT_STATIC=true ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "═══════════════════════════════════════════════════════════════"
log "Poker44 Miner-B Retrain Pipeline"
log "═══════════════════════════════════════════════════════════════"

# ─── Step 1: Download gold data ───
if [ "$SKIP_DOWNLOAD" = false ]; then
    log "Step 1: Downloading latest gold data..."
    python3 workspace/dataset/scripts/download_gold_data.py \
        --out-dir "$GOLD_DIR" \
        --max-days 0 \
        || log "WARNING: Gold download failed (API may be down)"
else
    log "Step 1: SKIPPED (--skip-download)"
fi

# ─── Step 2: Extract gold features ───
log "Step 2: Extracting gold features..."
python3 workspace/hybrid/scripts/extract_gold_features.py
log "  Gold features: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$GOLD_FEATURES')))" 2>/dev/null || echo 'N/A') rows"

# ─── Step 3: Extract static human features (zenodo/public) if not present ───
if [ "$SKIP_EXTRACT_STATIC" = false ]; then
    if [ ! -f "$ZENODO_FEATURES" ]; then
        log "Step 3a: Extracting zenodo features (first time)..."
        python3 workspace/hybrid/scripts/extract_zenodo_features.py
    else
        log "Step 3a: Zenodo features already exist, skipping"
    fi

    if [ ! -f "$PUBLIC_FEATURES" ]; then
        log "Step 3b: Extracting public features (first time)..."
        python3 workspace/hybrid/scripts/extract_public_features.py
    else
        log "Step 3b: Public features already exist, skipping"
    fi
else
    log "Step 3: SKIPPED (--skip-extract-static)"
fi

# ─── Step 4: Generate full-spectrum bots if not present ───
if [ "$SKIP_BOTS" = false ] && [ ! -f "$FULL_SPECTRUM" ]; then
    log "Step 4: Generating full-spectrum bot profiles (this takes a while)..."
    python3 workspace/hybrid/scripts/generate_full_spectrum_bots.py \
        --n-profiles 1000 \
        --chunks-per-profile 10
    log "  Full spectrum bots: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$FULL_SPECTRUM')))" 2>/dev/null || echo 'N/A') rows"
elif [ "$SKIP_BOTS" = true ]; then
    log "Step 4: SKIPPED (--skip-bots)"
else
    log "Step 4: Full-spectrum bots already exist, skipping"
fi

# ─── Step 5: Train production model ───
log "Step 5: Training production model..."
python3 workspace/hybrid/scripts/train_production_model.py \
    --output-dir "$MODEL_BUNDLE"

# ─── Step 6: Update .env for deployment ───
MODEL_PATH="$(realpath "$MODEL_BUNDLE/model.joblib")"
TRANSFORM_PATH="$(realpath "$MODEL_BUNDLE/transform_meta.json")"

log "Step 6: Model bundle ready at $MODEL_BUNDLE"
log ""
log "═══════════════════════════════════════════════════════════════"
log "DEPLOYMENT"
log "═══════════════════════════════════════════════════════════════"
log "To deploy Miner B (hybrid model), set in .env:"
log "  POKER44_MINER_MODEL_PATH=$MODEL_PATH"
log "  POKER44_MINER_TRANSFORM_META_PATH=$TRANSFORM_PATH"
log "  POKER44_MINER_OTHER_ONLY=0"
log "  POKER44_MINER_REQUIRE_MODEL=1"
log ""
log "To keep Miner A (other-only), keep:"
log "  POKER44_MINER_OTHER_ONLY=1"
log ""
log "Pipeline complete."
