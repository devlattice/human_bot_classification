#!/usr/bin/env bash
# Full feature selection v3 with live terminal output + timestamped log.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

LOG_DIR="workspace/hybrid/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/feature_selection_v3_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== $(date -Iseconds) feature selection v3 ==="
echo "log: $LOG_FILE"
echo "cwd: $(pwd)"
echo

python3 -u workspace/hybrid/scripts/extract_gold_features.py
python3 -u workspace/hybrid/scripts/split_gold_may8_to_test.py

python3 -u workspace/hybrid/scripts/feature_selection_v3.py --phase all \
  --trials 300 \
  --trials-per-x 50 \
  --top-save 20 \
  --top 10 \
  --x-base 30 \
  --borderline 25

echo
echo "=== done $(date -Iseconds) ==="
echo "winner: workspace/hybrid/selected_features_v3.json"
echo "report: workspace/hybrid/feature_selection_v3_report.txt"
