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
