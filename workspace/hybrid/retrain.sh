#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Poker44 Miner-B Daily Retrain Pipeline
# ═══════════════════════════════════════════════════════════════════
#
# This script automates the full retrain cycle:
#   1. Download latest gold data from benchmark API
#   2. Extract features from gold data
#   3. (Optional) Extract static / test features if missing
#   4. (Optional) Generate full-spectrum bot profiles if missing
#   4b. Optuna RF search (unless --skip-optuna)
#   5. Train production model
#   5b. PASS/FAIL gates from retrain_summary (unless --skip-gates)
#   6. KS analysis
#
# Usage:
#   ./workspace/hybrid/retrain.sh              # full pipeline
#   ./workspace/hybrid/retrain.sh --skip-bots  # skip bot generation
#   ./workspace/hybrid/retrain.sh --skip-download  # skip gold download
#   ./workspace/hybrid/retrain.sh --skip-optuna  # skip RF hyperparameter search
#   ./workspace/hybrid/retrain.sh --skip-gates   # skip PASS/FAIL gate check
#   ./workspace/hybrid/retrain.sh --force-reextract  # drop cached parquets after payload_view change
#
# Optuna trial count (default 40):  OPTUNA_N_TRIALS=80 ./workspace/hybrid/retrain.sh
#
# Gate thresholds (percent points @0.5), after training:
#   RETRAIN_GATE_MIN_MAY8_RECALL_PCT=80 RETRAIN_GATE_MAX_ZENODO_FPR_PCT=2.0 ...
# Fail the whole script on gate failure (e.g. CI):
#   RETRAIN_GATES_STRICT=1 ./workspace/hybrid/retrain.sh
#
# Cron example (daily at 2 AM UTC):
#   0 2 * * * cd /home/dr/Workspace/Poker44-subnet && ./workspace/hybrid/retrain.sh >> workspace/hybrid/retrain.log 2>&1
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GOLD_DIR="workspace/dataset/source/gold_dataset"
TRAIN_DIR="workspace/hybrid/dataset/train"
TEST_DIR="workspace/hybrid/dataset/test"
GOLD_FEATURES="$TRAIN_DIR/gold_features.parquet"
ZENODO_FEATURES="$TRAIN_DIR/zenodo_features.parquet"
PUBLIC_FEATURES="$TRAIN_DIR/public_features.parquet"
ACPC_BOT_FEATURES="$TRAIN_DIR/acpc_bot_features.parquet"
FULL_SPECTRUM="$TRAIN_DIR/full_spectrum_bot_features.parquet"
MODEL_BUNDLE="workspace/hybrid/model_bundle"
KS_DIR="workspace/hybrid/KS_test"
ENV_FILE=".env"

SKIP_BOTS=false
SKIP_DOWNLOAD=false
SKIP_EXTRACT_STATIC=false
SKIP_OPTUNA=false
SKIP_GATES=false
FORCE_REEXTRACT=false

FEATURE_PIPELINE_STAMP="workspace/hybrid/.feature_pipeline_version"
# Keep in sync with workspace/hybrid/chunk_pipeline.py FEATURE_PIPELINE_VERSION
FEATURE_PIPELINE_EXPECTED="payload-view-action-leak-tighten-2026-05"

for arg in "$@"; do
    case "$arg" in
        --skip-bots) SKIP_BOTS=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-extract-static) SKIP_EXTRACT_STATIC=true ;;
        --skip-optuna) SKIP_OPTUNA=true ;;
        --skip-gates) SKIP_GATES=true ;;
        --force-reextract) FORCE_REEXTRACT=true ;;
    esac
done

invalidate_stale_feature_parquets() {
    local current=""
    if [ -f "$FEATURE_PIPELINE_STAMP" ]; then
        current=$(cat "$FEATURE_PIPELINE_STAMP")
    fi
    if [ "$FORCE_REEXTRACT" = true ] || [ "$current" != "$FEATURE_PIPELINE_EXPECTED" ]; then
        log "Feature pipeline version change ($current -> $FEATURE_PIPELINE_EXPECTED); removing cached parquets..."
        rm -f "$ZENODO_FEATURES" "$PUBLIC_FEATURES" "$ACPC_BOT_FEATURES"
        rm -f "$TEST_DIR/zenodo_test_features.parquet"
        rm -f "$TEST_DIR/public_test_features.parquet"
        rm -f "$TEST_DIR/acpc_bot_test_features.parquet"
        rm -f "$FULL_SPECTRUM"
        rm -f "$TRAIN_DIR/generated_bot_features.parquet"
        rm -f "$TRAIN_DIR/calibrated_bot_features.parquet"
        mkdir -p "$(dirname "$FEATURE_PIPELINE_STAMP")"
        echo "$FEATURE_PIPELINE_EXPECTED" > "$FEATURE_PIPELINE_STAMP"
    fi
}

OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-40}"
RETRAIN_GATE_MIN_MAY8_RECALL_PCT="${RETRAIN_GATE_MIN_MAY8_RECALL_PCT:-80}"
RETRAIN_GATE_MAX_ZENODO_FPR_PCT="${RETRAIN_GATE_MAX_ZENODO_FPR_PCT:-2.0}"
RETRAIN_GATE_MAX_PUBLIC_FPR_PCT="${RETRAIN_GATE_MAX_PUBLIC_FPR_PCT:-3.0}"
RETRAIN_GATE_MIN_ACPC_RECALL_PCT="${RETRAIN_GATE_MIN_ACPC_RECALL_PCT:-90}"
RETRAIN_GATES_STRICT="${RETRAIN_GATES_STRICT:-0}"
BEST_RF_JSON="$MODEL_BUNDLE/best_rf_params.json"
EXTRA_RF_ARGS=()

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

invalidate_stale_feature_parquets

# ─── Step 3: Extract static features (zenodo/public/acpc) if not present ───
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

    if [ ! -f "$ACPC_BOT_FEATURES" ]; then
        log "Step 3c: Extracting ACPC bot features (first time)..."
        python3 workspace/hybrid/scripts/extract_acpc_bot_features.py
    else
        log "Step 3c: ACPC bot features already exist, skipping"
    fi

    # Test sets (only need to extract once)
    if [ ! -f "$TEST_DIR/zenodo_test_features.parquet" ]; then
        log "Step 3d: Extracting test set features (first time)..."
        python3 workspace/hybrid/scripts/extract_test_features.py
    else
        log "Step 3d: Test set features already exist, skipping"
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

# ─── Step 4b: Optuna RF search (writes best_rf_params.json for training) ───
mkdir -p "$MODEL_BUNDLE"
if [ "$SKIP_OPTUNA" = false ]; then
    log "Step 4b: Optuna RandomForest search ($OPTUNA_N_TRIALS trials)..."
    if python3 workspace/hybrid/scripts/optuna_tune_rf.py \
        --out-json "$BEST_RF_JSON" \
        --n-trials "$OPTUNA_N_TRIALS"; then
        log "  Optuna finished; best params → $BEST_RF_JSON"
    else
        log "WARNING: Optuna step failed (missing optuna, data, or error). Training uses CLI defaults unless $BEST_RF_JSON already exists."
    fi
else
    log "Step 4b: SKIPPED (--skip-optuna)"
fi
if [ -f "$BEST_RF_JSON" ]; then
    EXTRA_RF_ARGS+=(--rf-params-json "$BEST_RF_JSON")
    log "  Training will apply RF patch from $BEST_RF_JSON"
fi

# ─── Step 5: Train production model ───
log "Step 5: Training production model..."
# CLI defaults apply for keys not in best_rf_params.json; override e.g.:
#   EXTRA_RF_ARGS=(--rf-n-estimators 400)  or edit JSON after Optuna
python3 workspace/hybrid/scripts/train_production_model.py \
    "${EXTRA_RF_ARGS[@]}" \
    --output-dir "$MODEL_BUNDLE"

# ─── Step 5b: PASS/FAIL gates (reads retrain_summary.json) ───
if [ "$SKIP_GATES" = false ]; then
    log "Step 5b: Checking retrain gates..."
    _gate_cmd=(python3 workspace/hybrid/scripts/check_retrain_gates.py
        --summary "$MODEL_BUNDLE/retrain_summary.json"
        --min-may8-recall-pct "$RETRAIN_GATE_MIN_MAY8_RECALL_PCT"
        --max-zenodo-fpr-pct "$RETRAIN_GATE_MAX_ZENODO_FPR_PCT"
        --max-public-fpr-pct "$RETRAIN_GATE_MAX_PUBLIC_FPR_PCT"
        --min-acpc-recall-pct "$RETRAIN_GATE_MIN_ACPC_RECALL_PCT")
    if "${_gate_cmd[@]}"; then
        log "  Retrain gates: PASS"
    else
        log "  Retrain gates: FAIL (see block above)"
        if [ "$RETRAIN_GATES_STRICT" = "1" ]; then
            log "ERROR: RETRAIN_GATES_STRICT=1 — aborting pipeline."
            exit 1
        fi
    fi
else
    log "Step 5b: SKIPPED (--skip-gates)"
fi

# ─── Step 6: KS analysis on selected features ───
log "Step 6: Running KS analysis..."
python3 workspace/hybrid/scripts/ks_analysis.py || log "WARNING: KS analysis failed (non-critical)"
log "  KS results: $KS_DIR/"

# ─── Step 7: Update .env for deployment ───
MODEL_PATH="$(realpath "$MODEL_BUNDLE/model.joblib")"
TRANSFORM_PATH="$(realpath "$MODEL_BUNDLE/transform_meta.json")"

log "Step 7: Model bundle ready at $MODEL_BUNDLE"
log ""
log "═══════════════════════════════════════════════════════════════"
log "DEPLOYMENT"
log "═══════════════════════════════════════════════════════════════"
log "To deploy Miner B (54-feat daily bundle), set in .env:"
log "  POKER44_MINER_MODEL_PATH=$MODEL_PATH"
log "  POKER44_MINER_TRANSFORM_META_PATH=$TRANSFORM_PATH"
log "  POKER44_MINER_OTHER_ONLY=0"
log "  POKER44_MINER_REQUIRE_MODEL=1"
log ""
log "V11 production (11 features, static @0.55) — use workspace/model/artifacts/:"
log "  ./workspace/model/deploy_v11_prod.sh"
log ""
log "To keep Miner A (other-only), keep:"
log "  POKER44_MINER_OTHER_ONLY=1"
log ""
log "Pipeline complete."
