#!/usr/bin/env bash
set -euo pipefail

# Quick A/B/C mini-grid runner for uncertain-band params.
# Produces per-run cross eval outputs and one merged summary table.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-workspace/model/artifacts/lgbm_2_v1/lgbm_classifier.joblib}"
THRESHOLD="${THRESHOLD:-0.5}"
SELECTED_THRESHOLD="${SELECTED_THRESHOLD:-0.952}"
OUT_BASE="${OUT_BASE:-workspace/test/parameter_test}"

EVAL_1="${EVAL_1:-workspace/dataset/robusted_dataset/original_test/hollout_human_mix.parquet}"
EVAL_2="${EVAL_2:-workspace/dataset/robusted_dataset/original_test/hollout_train.parquet}"
EVAL_3="${EVAL_3:-workspace/dataset/robusted_dataset/original_test/hollout_test.parquet}"

# Resolve output base: if not writable, fallback to /tmp.
if ! mkdir -p "$OUT_BASE" 2>/dev/null || ! ( : > "$OUT_BASE/.write_test" ) 2>/dev/null; then
  OUT_BASE="${TMPDIR:-/tmp}/poker44_parameter_test"
  mkdir -p "$OUT_BASE"
  echo "[grid] warning: default output dir not writable; using fallback: $OUT_BASE"
else
  rm -f "$OUT_BASE/.write_test"
fi

run_one() {
  local run_id="$1"
  local a="$2"
  local b="$3"
  local g="$4"
  local out_dir="$OUT_BASE/run_${run_id}"

  echo "[grid] run=${run_id} a=${a} b=${b} gamma=${g}"
  export POKER44_MINER_UNCERTAIN_A="$a"
  export POKER44_MINER_UNCERTAIN_B="$b"
  export POKER44_MINER_UNCERTAIN_GAMMA="$g"

  mkdir -p "$out_dir"
  PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
    --model "$MODEL_PATH" \
    --eval-parquet "$EVAL_1" \
    --eval-parquet "$EVAL_2" \
    --eval-parquet "$EVAL_3" \
    --threshold "$THRESHOLD" \
    --selected-threshold "$SELECTED_THRESHOLD" \
    --out-dir "$out_dir"
}

run_one "A" "0.3" "0.7" "1.4"
run_one "B" "0.4" "0.6" "1.4"
run_one "C" "-1" "-1" "1.0"

OUT_BASE="$OUT_BASE" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

base = Path(os.environ["OUT_BASE"])
runs = [
    ("A", 0.3, 0.7, 1.4, "yes"),
    ("B", 0.4, 0.6, 1.4, "yes"),
    ("C", -1.0, -1.0, 1.0, "no"),
]
rows = []
for run_id, a, b, g, smoothing_on in runs:
    p = base / f"run_{run_id}" / "cross_dataset_comparison.csv"
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "a": a,
                "b": b,
                "gamma": g,
                "smoothing_on": smoothing_on,
                "dataset": Path(str(r["dataset_dir"])).name,
                "threshold_used": r.get("threshold_used"),
                "selected_threshold": r.get("selected_threshold"),
                "val_human_fpr": r.get("val_human_fpr"),
                "val_bot_recall": r.get("val_bot_recall"),
                "val_accuracy": r.get("val_accuracy"),
                "val_human_fpr_at_selected": r.get("val_human_fpr_at_selected"),
                "val_bot_recall_at_selected": r.get("val_bot_recall_at_selected"),
            }
        )

out = pd.DataFrame(rows)
out_path = base / "grid_summary.csv"
out.to_csv(out_path, index=False)
print(f"[grid] wrote {out_path}")
PY

echo "[grid] done. See:"
echo "  - $OUT_BASE/run_A/cross_dataset_comparison.csv"
echo "  - $OUT_BASE/run_B/cross_dataset_comparison.csv"
echo "  - $OUT_BASE/run_C/cross_dataset_comparison.csv"
echo "  - $OUT_BASE/grid_summary.csv"
