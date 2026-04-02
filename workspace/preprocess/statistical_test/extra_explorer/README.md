# Extra explorer — domain-shift experiment system

Small workspace to run **incremental** experiments: reduce train vs validator **marginal shift** without silently destroying **human vs bot** signal.

## What lives here

| File | Purpose |
|------|---------|
| `EXPERIMENT_SHEET.md` | Human-readable rules, pass/fail checklist, dummy example |
| `experiments_template.csv` | Column headers for a spreadsheet log (copy → `experiments.csv`, gitignore if needed) |
| `collect_experiment_metrics.py` | One command to summarize **shift CSV** + optional **cross_eval CSV** into JSON (paste into sheet or append) |

## End-to-end loop (one experiment)

1. **Pick one change only** (harmonization *or* stronger clip *or* regularization *or* 2–5 feature ablation).
2. **Rebuild** robust train/val + `transform_meta.json` if preprocessing changed.
3. **Retrain** model → new artifact path.
4. **Shift:** run `train_validator_shift_plots.py` with **fixed** train parquet + **fixed** validator robusted parquet → get `train_vs_validator_shift.csv`.
5. **Task:** run `cross_dataset_eval.py` on the same model → get `cross_dataset_comparison.csv`.
6. **Collect metrics:**

```bash
cd /path/to/Poker44-subnet
PYTHONPATH=. python workspace/preprocess/statistical_test/extra_explorer/collect_experiment_metrics.py \
  --run-id E1_harmonization_v1 \
  --shift-csv workspace/preprocess/statistical_test/plots/miner_2/train_vs_validator_robusted/train_vs_validator_shift.csv \
  --cross-eval-csv workspace/model/artifacts/cross_eval/your_run/cross_dataset_comparison.csv \
  --notes "rank-gauss after robust; q-low 0.02"
```

7. **Decide** using `EXPERIMENT_SHEET.md` pass/fail vs baseline `E0`.

## Planned experiments (suggested order)

- **E0** Baseline — current 44 + current robust meta + current LGBM.
- **E1** Marginal harmonization — train-fitted quantile/rank mapping **after** robust transform (when implemented); re-run shift + cross_eval.
- **E2** Stronger robust knobs — e.g. `--q-low 0.02 --q-high 0.98 --scaled-clip-abs 6`; refit meta; retrain.
- **E3** Small ablation — drop 2–5 features that are high-shift **and** weak in ablation; retrain.

## Principles

- Same validator parquet window across runs when possible.
- Never adopt on **KS alone** — always check holdout / cross_eval columns you care about.
- Log every run in `experiments.csv` (or your sheet) so you can diff baselines.

## Related repo paths

- Shift plots: `workspace/preprocess/statistical_test/train_validator_shift_plots.py`
- Cross-eval: `workspace/test/cross_dataset_eval.py`
- Robust fit/apply: `workspace/preprocess/robust_feature_transform.py`
