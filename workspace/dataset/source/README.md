# Subnet target — dataset source

Human hand histories live in `poker_hands_combined.json` (a JSON **array** of hand objects). Use `source_split.py` to shuffle and split them for train/test before mixing with bot data or downstream preprocessing.

## Train / test split

Run from this directory (or pass absolute paths):

```bash
python3 source_split.py
```

Defaults:

| Input | `poker_hands_combined.json` |
| Train output | `poker_hands_train.json` |
| Test output | `poker_hands_test.json` |
| Ratio | 80% train / 20% test (`--train-ratio 0.8`) |
| Shuffle | Reproducible with `--seed 42` |

Override as needed:

```bash
python3 source_split.py --input poker_hands_combined.json \
  --train-out poker_hands_train.json --test-out poker_hands_test.json \
  --train-ratio 0.8 --seed 42
```

The script shuffles **indices**, then takes the first `round(n * train_ratio)` hands for training and the rest for test. Order inside each file follows that shuffled order.

## Holdout human pools (evaluation)

| File | Role |
|------|------|
| `poker_hands_test.json` | **Subnet public human** holdout: the test side of the 80/20 split from `poker_hands_combined.json` (same hand schema). Use as unseen human traffic *relative to* `poker_hands_train.json` — do not train on this file if you want a clean public-human eval. |
| `zenodo_holdout.json` | **Zenodo human** holdout: hands reserved outside the Zenodo pool used for training / `hollout_test`-style mixes. Use for unseen Zenodo-style humans (orthogonal to subnet public split). |

When building mixed human+bot eval parquets, point `workspace/preprocess/build_dataset.py --human-json` at the appropriate file, then apply the same frozen `transform_meta.json` + keep-list as training (`workspace/preprocess/robust_feature_transform.py` apply-only mode).
