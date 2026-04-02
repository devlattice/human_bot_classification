# Domain Shift Experiment Sheet (Simple)

Use this sheet to decide if a new run is better than the current baseline.

## Baseline (fill once)

- Model/run id:
- Feature set:
- Transform config:
- Threshold policy:
- Notes:

### Baseline metrics

- Shift side (train vs validator logs):
  - `max_ks`:
  - `count_ks_ge_0_20`:
  - `count_ks_ge_0_15`:
  - `n_sig_fdr_0_05`:
- Task side (cross-eval / holdout):
  - `val_roc_auc`:
  - `val_human_fpr_at_selected`:
  - `val_bot_recall_at_selected`:
  - `val_reward`:

---

## Pass / Fail Rules

Adopt a candidate only if **all** are true:

1. Shift improves:
   - `max_ks` decreases, and
   - `count_ks_ge_0_20` does not increase.
2. Task quality is preserved:
   - `val_roc_auc` drop <= `0.005` (absolute), and
   - `val_human_fpr_at_selected` does not increase, and
   - `val_bot_recall_at_selected` does not drop by more than `0.02`.
3. Reward side is not worse for your operating mode:
   - `val_reward` does not decrease materially (your threshold: ____).

If not, reject and keep baseline.

---

## Candidate Runs

| run_id | change_type | key_args_or_feature_changes | max_ks | ks>=0.20 | sig_fdr | val_roc_auc | val_human_fpr_at_selected | val_bot_recall_at_selected | val_reward | pass/fail | reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| E0 | baseline | current 44 features + current transform |  |  |  |  |  |  |  |  |  |
| E1 | harmonization | train-fitted rank/quantile mapping after robust transform |  |  |  |  |  |  |  |  |  |
| E2 | regularization | stronger LGBM regularization, no feature drop |  |  |  |  |  |  |  |  |  |
| E3 | small ablation | drop top 2-5 high-shift weak-signal features |  |  |  |  |  |  |  |  |  |

---

## Dummy Example (how to decide)

- Baseline: `max_ks=0.26`, `ks>=0.20=9`, `auc=0.992`, `fpr_sel=0.00`, `recall_sel=0.85`
- Candidate A: `max_ks=0.22`, `ks>=0.20=6`, `auc=0.991`, `fpr_sel=0.00`, `recall_sel=0.84`
  - Decision: **PASS** (shift improved, task drop small)
- Candidate B: `max_ks=0.20`, `ks>=0.20=4`, `auc=0.978`, `fpr_sel=0.01`, `recall_sel=0.80`
  - Decision: **FAIL** (task drop too large + safety worse)

---

## Quick Notes

- Compare runs on the **same validator log window** when possible.
- Do not pick based on KS only.
- Keep changes small and incremental.
