#!/bin/bash

# Poker44 Miner Startup Script

# Optional env-file loading:
# - default: <repo-root>/.env (works from any cwd)
# - override: MINER_ENV_FILE=/path/to/file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MINER_ENV_FILE="${MINER_ENV_FILE:-$REPO_ROOT/.env}"
if [ -f "$MINER_ENV_FILE" ]; then
  echo "Loading miner env from: $MINER_ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$MINER_ENV_FILE"
  set +a
fi

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-rowan_wallet}"
HOTKEY="${HOTKEY:-poker_miner_1}"
NETWORK="${NETWORK:-finney}" ##finney
# Resolved after cd to REPO_ROOT (absolute path used for PM2).
MINER_SCRIPT="${MINER_SCRIPT:-neurons/miner.py}"
PM2_NAME="${PM2_NAME:-poker_miner_1}"  ##  name of Miner, as you wish
AXON_PORT="${AXON_PORT:-8080}"
# Space-separated SS58 validator *hotkeys* (not coldkey, not wallet name).
# Must match ``synapse.dendrite.hotkey`` exactly. Strip CR from Windows .env lines.
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS//$'\r'/}"

if [ "$WALLET_NAME" = "your_wallet" ] || [ "$HOTKEY" = "your_hotkey" ]; then
    echo "Error: WALLET_NAME/HOTKEY still use placeholders. Update .env before starting."
    exit 1
fi

if [ -z "$NETWORK" ]; then
    echo "Error: NETWORK is empty. Set NETWORK=finney (or your target network)."
    exit 1
fi

if [ -z "$AXON_PORT" ]; then
    echo "Error: AXON_PORT is empty. Set AXON_PORT in .env (e.g., 8091)."
    exit 1
fi

cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT"

# joblib LightGBM loads need ``import lightgbm`` in the SAME interpreter PM2 uses.
# Interpreter selection priority:
# 1) MINER_PYTHON (explicit python binary)
# 2) SHARED_MINER_VENV/bin/python (shared venv for many miners)
# 3) <repo>/.venv/bin/python (repo-local venv)
# 4) system python3/python
MINER_PYTHON="${MINER_PYTHON:-}"
SHARED_MINER_VENV="${SHARED_MINER_VENV:-}"
if [ -z "$MINER_PYTHON" ] && [ -n "$SHARED_MINER_VENV" ] && [ -x "$SHARED_MINER_VENV/bin/python" ]; then
  MINER_PYTHON="$SHARED_MINER_VENV/bin/python"
fi
if [ -z "$MINER_PYTHON" ] && [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  MINER_PYTHON="$REPO_ROOT/.venv/bin/python"
fi
if [ -z "$MINER_PYTHON" ]; then
  MINER_PYTHON="$(command -v python3 || command -v python || true)"
fi
if [ -z "$MINER_PYTHON" ] || [ ! -x "$MINER_PYTHON" ]; then
  echo "Error: no Python interpreter found. Set MINER_PYTHON, or SHARED_MINER_VENV, or create $REPO_ROOT/.venv"
  exit 1
fi
if ! "$MINER_PYTHON" -c "import lightgbm" 2>/dev/null; then
  echo "Error: $MINER_PYTHON cannot import lightgbm — model .joblib will not load."
  echo "Install: $MINER_PYTHON -m pip install lightgbm"
  exit 1
fi
if [ "$NETWORK" = "local" ] && [ -z "${BT_AXON_EXTERNAL_IP:-}" ]; then
  BT_AXON_EXTERNAL_IP="$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i !~ /^127\./) {print $i; exit}}')"
  export BT_AXON_EXTERNAL_IP="${BT_AXON_EXTERNAL_IP:-127.0.0.1}"
fi

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  # Trim whitespace-only entries
  _trimmed=()
  for _hk in "${VALIDATOR_HOTKEY_ARRAY[@]}"; do
    _t="${_hk#"${_hk%%[![:space:]]*}"}"
    _t="${_t%"${_t##*[![:space:]]}"}"
    if [ -n "$_t" ]; then
      _trimmed+=("$_t")
    fi
  done
  VALIDATOR_HOTKEY_ARRAY=("${_trimmed[@]}")
  if [ "${#VALIDATOR_HOTKEY_ARRAY[@]}" -eq 0 ]; then
    echo "Error: ALLOWED_VALIDATOR_HOTKEYS is set but no hotkeys parsed (check quoting)."
    exit 1
  fi
  echo "Validator allowlist (${#VALIDATOR_HOTKEY_ARRAY[@]} keys): ${VALIDATOR_HOTKEY_ARRAY[*]}"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

# Use absolute script path so PM2 cwd does not matter
case "$MINER_SCRIPT" in
  /*) MINER_SCRIPT_ABS="$MINER_SCRIPT" ;;
  *)  MINER_SCRIPT_ABS="$REPO_ROOT/$MINER_SCRIPT" ;;
esac
if [ ! -f "$MINER_SCRIPT_ABS" ]; then
  echo "Error: Miner script not found at $MINER_SCRIPT_ABS"
  exit 1
fi

pm2 start "$MINER_PYTHON" \
  --name "$PM2_NAME" -- \
  "$MINER_SCRIPT_ABS" \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME (python=$MINER_PYTHON)"
echo "View logs: pm2 logs $PM2_NAME"
