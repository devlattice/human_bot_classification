#!/bin/bash

set -euo pipefail

# Poker44 Validator Startup Script

NETUID=126
WALLET_NAME="poker44-test-ck"
HOTKEY="poker44-hk"
NETWORK="finney"
VALIDATOR_SCRIPT="./neurons/validator.py"
PM2_NAME="poker44_validator"  ##  name of validator, as you wish
POKER44_HUMAN_JSON_PATH="/path/to/private/poker_data_combined.json"
POKER44_VALIDATOR_SECRET_KEY="shared-secret-for-sn126"
POKER44_CHUNK_COUNT=40
POKER44_REWARD_WINDOW=40
POKER44_POLL_INTERVAL_SECONDS=300
NEURON_TIMEOUT=60

if [ ! -f "$VALIDATOR_SCRIPT" ]; then
    echo "Error: Validator script not found at $VALIDATOR_SCRIPT"
    exit 1
fi

if [ ! -f "$POKER44_HUMAN_JSON_PATH" ]; then
    echo "Error: Private validator human dataset not found at $POKER44_HUMAN_JSON_PATH"
    echo "Set POKER44_HUMAN_JSON_PATH in scripts/validator/run/run_vali.sh before starting."
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import bittensor, dotenv, numpy, pandas, sklearn" >/dev/null 2>&1; then
    echo "Error: Python environment is missing required packages for validator startup."
    echo "Checked interpreter: $PYTHON_BIN"
    echo "Run ./scripts/validator/main/setup.sh or fix the virtualenv before starting PM2."
    exit 1
fi

GIT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"
DEPLOY_VERSION="$(grep -E '^VALIDATOR_DEPLOY_VERSION[[:space:]]*=' poker44/__init__.py 2>/dev/null | head -n1 | sed -E 's/^VALIDATOR_DEPLOY_VERSION[[:space:]]*=[[:space:]]*["'\'']([^"'\'']+)["'\'']/\1/')"

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
export POKER44_HUMAN_JSON_PATH="$POKER44_HUMAN_JSON_PATH"
export POKER44_CHUNK_COUNT="$POKER44_CHUNK_COUNT"
export POKER44_REWARD_WINDOW="$POKER44_REWARD_WINDOW"
export POKER44_POLL_INTERVAL_SECONDS="$POKER44_POLL_INTERVAL_SECONDS"

pm2 start $VALIDATOR_SCRIPT \
  --name $PM2_NAME -- \
  "${VALIDATOR_ARGS[@]}"

pm2 save

echo "Validator started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Code: branch=${GIT_BRANCH:-<unknown>} commit=${GIT_COMMIT:-<unknown>} deploy_version=${DEPLOY_VERSION:-<unknown>}"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY python=$PYTHON_BIN"
echo "Subtensor args: ${SUBTENSOR_PARAM:---subtensor.network $NETWORK}"
echo "Runtime extras: wallet_path=${WALLET_PATH:-<default>} extra_args=${VALIDATOR_EXTRA_ARGS:-<none>}"
echo "Profile: chunks=$POKER44_CHUNK_COUNT reward_window=$POKER44_REWARD_WINDOW poll_interval_s=$POKER44_POLL_INTERVAL_SECONDS miners_per_cycle=$POKER44_MINERS_PER_CYCLE timeout_s=$NEURON_TIMEOUT sync_window_mode=$POKER44_SYNCED_WINDOW_MODE sync_all_miners=$POKER44_SYNC_ALL_MINERS direct_score_update=$POKER44_SYNC_DIRECT_SCORE_UPDATE"
