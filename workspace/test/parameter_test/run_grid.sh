#!/usr/bin/env bash
set -euo pipefail

# Uncertain-band (a, b, gamma) grid runner.
# Default grid: A–F (see run_one calls below). Merge step scans run_*/cross_dataset_comparison.csv.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# Default model: first path that exists, else legacy default (override with MODEL_PATH=...).
if [[ -z "${MODEL_PATH:-}" ]]; then
  for _cand in \
    "workspace/model/artifacts/lgbm_weak_ssl_ultralow_ssl_weak_mcp055/lgbm_b_classifier_calibrated.joblib" \
    "workspace/model/artifacts/lgbm_2_v1/lgbm_classifier.joblib"; do
    if [[ -f "$_cand" ]]; then
      MODEL_PATH="$_cand"
      break
    fi
  done
  MODEL_PATH="${MODEL_PATH:-workspace/model/artifacts/lgbm_2_v1/lgbm_classifier.joblib}"
fi
echo "[grid] MODEL_PATH=${MODEL_PATH}"
THRESHOLD="${THRESHOLD:-0.5}"
SELECTED_THRESHOLD="${SELECTED_THRESHOLD:-0.952}"
OUT_BASE="${OUT_BASE:-workspace/test/parameter_test}"

EVAL_1="${EVAL_1:-workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_1.parquet}"
EVAL_2="${EVAL_2:-workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/pb_2.parquet}"
EVAL_3="${EVAL_3:-workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_1.parquet}"
EVAL_4="${EVAL_4:-workspace/preprocess/statistical_test/explorer/miner_1/feature_1/data/test/holdout_2.parquet}"

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
  # shellcheck disable=SC2206
  local eval_args=(
    --eval-parquet "$EVAL_1"
    --eval-parquet "$EVAL_2"
    --eval-parquet "$EVAL_3"
  )
  if [[ -f "$EVAL_4" ]]; then
    eval_args+=(--eval-parquet "$EVAL_4")
  fi

  PYTHONPATH=. python workspace/test/cross_dataset_eval.py \
    --model "$MODEL_PATH" \
    "${eval_args[@]}" \
    --threshold "$THRESHOLD" \
    --selected-threshold "$SELECTED_THRESHOLD" \
    --uncertain-a "$a" \
    --uncertain-b "$b" \
    --uncertain-gamma "$g" \
    --out-dir "$out_dir"
}

# A–F: same semantics as miner (F = smoothing off via invalid a,b).
run_one "A" "0.3" "0.7" "1.4"
run_one "B" "0.3" "0.7" "1.2"
run_one "C" "0.4" "0.6" "1.4"
run_one "D" "0.2" "0.8" "1"
run_one "E" "0.4" "0.55" "0.9"
run_one "F" "-1" "-1" "1.0"

OUT_BASE="$OUT_BASE" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

base = Path(os.environ["OUT_BASE"])
paths = sorted(base.glob("run_*/cross_dataset_comparison.csv"))
if not paths:
    raise SystemExit(f"[grid] no run_*/cross_dataset_comparison.csv under {base}")

rows = []
for p in paths:
    run_id = p.parent.name.removeprefix("run_")
    df = pd.read_csv(p)
    if df.empty:
        continue
    r0 = df.iloc[0]
    a = float(r0["uncertain_a"])
    b = float(r0["uncertain_b"])
    g = float(r0["uncertain_gamma"])
    smoothing_on = (
        "yes"
        if (0.0 <= a < b <= 1.0 and g > 0.0)
        else "no"
    )
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
print(f"[grid] wrote {out_path} ({len(paths)} run dir(s))")
PY

echo "[grid] done. See: $OUT_BASE/run_{A..F}/cross_dataset_comparison.csv and $OUT_BASE/grid_summary.csv"
