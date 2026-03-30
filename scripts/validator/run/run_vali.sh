#!/bin/bash

set -euo pipefail

# Poker44 Validator Startup Script

NETUID="${NETUID:-1}"
WALLET_NAME="${WALLET_NAME:-validator}"
HOTKEY="${HOTKEY:-default}"
NETWORK="${NETWORK:-local}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-./neurons/validator.py}"
PM2_NAME="${PM2_NAME:-poker_validator}"  ##  name of validator, as you wish
VALIDATOR_ENV_DIR="${VALIDATOR_ENV_DIR:-validator_env}"
WALLET_PATH="${WALLET_PATH:-}"
VALIDATOR_EXTRA_ARGS="${VALIDATOR_EXTRA_ARGS:-}"
POKER44_HUMAN_JSON_PATH="${POKER44_HUMAN_JSON_PATH:-././hands_generator/human_hands/poker_hands_combined.json}"
POKER44_CHUNK_COUNT="${POKER44_CHUNK_COUNT:-40}"
POKER44_REWARD_WINDOW="${POKER44_REWARD_WINDOW:-40}"
POKER44_POLL_INTERVAL_SECONDS="${POKER44_POLL_INTERVAL_SECONDS:-60}"
POKER44_MINERS_PER_CYCLE="${POKER44_MINERS_PER_CYCLE:-16}"
NEURON_TIMEOUT="${NEURON_TIMEOUT:-60}"
VALIDATOR_AXON_PORT="${VALIDATOR_AXON_PORT:-8092}"

if [ -x "$VALIDATOR_ENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VALIDATOR_ENV_DIR/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: No Python interpreter found"
    exit 1
fi

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

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
if [ "$NETWORK" = "local" ] && [ -z "${BT_AXON_EXTERNAL_IP:-}" ]; then
  BT_AXON_EXTERNAL_IP="$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i !~ /^127\./) {print $i; exit}}')"
  export BT_AXON_EXTERNAL_IP="${BT_AXON_EXTERNAL_IP:-127.0.0.1}"
fi
export POKER44_HUMAN_JSON_PATH="$POKER44_HUMAN_JSON_PATH"
export POKER44_CHUNK_COUNT="$POKER44_CHUNK_COUNT"
export POKER44_REWARD_WINDOW="$POKER44_REWARD_WINDOW"
export POKER44_POLL_INTERVAL_SECONDS="$POKER44_POLL_INTERVAL_SECONDS"
export POKER44_MINERS_PER_CYCLE="$POKER44_MINERS_PER_CYCLE"

VALIDATOR_ARGS=(
  "$VALIDATOR_SCRIPT"
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$VALIDATOR_AXON_PORT"
  --neuron.timeout "$NEURON_TIMEOUT"
  --logging.debug
)

if [ -n "$WALLET_PATH" ]; then
  VALIDATOR_ARGS+=(--wallet.path "$WALLET_PATH")
fi

if [ -n "$VALIDATOR_EXTRA_ARGS" ]; then
  read -r -a EXTRA_ARG_ARRAY <<< "$VALIDATOR_EXTRA_ARGS"
  VALIDATOR_ARGS+=("${EXTRA_ARG_ARRAY[@]}")
fi

pm2 start "$PYTHON_BIN" \
  --name $PM2_NAME -- \
  "${VALIDATOR_ARGS[@]}"

pm2 save

echo "Validator started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY python=$PYTHON_BIN"
echo "Runtime extras: wallet_path=${WALLET_PATH:-<default>} extra_args=${VALIDATOR_EXTRA_ARGS:-<none>}"
echo "Profile: chunks=$POKER44_CHUNK_COUNT reward_window=$POKER44_REWARD_WINDOW poll_interval_s=$POKER44_POLL_INTERVAL_SECONDS miners_per_cycle=$POKER44_MINERS_PER_CYCLE timeout_s=$NEURON_TIMEOUT axon_port=$VALIDATOR_AXON_PORT"
