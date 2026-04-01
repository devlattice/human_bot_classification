# Subnet target — preprocess (`build_dataset.py`)

Builds **LightGBM-style** training data: mixed **human + bot** labeled chunks, per-hand sanitization, chunk-level features, then **train/val** Parquet (optionally **sharded** + `--resume`).

This entry point uses **subnet defaults** for human JSON and output directory. The same logic also lives under `workspace/datasets/_preprocess/preprocess_lightgbm.py` with **workspace-wide** defaults if you need that layout.

## Prerequisites

From repo root:

```bash
pip install -r workspace/datasets/requirements-dataset.txt
```

## Human source (single pool)

Default human hands:

`workspace/_subnet_target/dataset/source/poker_hands_train.json`

Produce it with `workspace/_subnet_target/dataset/source/source_split.py` from `poker_hands_combined.json` (8:2 shuffle split).

**Zenodo / second corpus is optional.** Only if you pass **both** `--human-json-a` and `--human-json-b` does the pipeline merge two pools (`_balanced_human_pool.json`). For subnet **original human + bot** only, omit those flags.

## Defaults (subnet)

| Flag | Default |
|------|---------|
| `--human-json` | `workspace/_subnet_target/dataset/source/poker_hands_train.json` |
| `--out` | `workspace/_subnet_target/dataset/unpreprocessed/train` |

Override either flag as needed. Other common flags: `--preset training-merged`, `--chunk-count`, `--shard-size`, `--resume`, `--bot-profile-mode mixed`, `--seed`.

### Directory permissions

If `workspace/_subnet_target/dataset/` was created as **root** (e.g. Docker or `sudo`), your user cannot create `unpreprocessed/…` and you will get `PermissionError`. Fix ownership, then rerun:

```bash
sudo chown -R "$USER:$USER" workspace/_subnet_target/dataset
```

Or write somewhere you own, e.g. `--out "$HOME/poker44_subnet_output/train"`.

Note: the default path uses **`unpreprocessed`** (full spelling). If you only have a typo folder `unpreprocesed`, either rename it or pass `--out` explicitly.

## Example (minimal)

From repo root, after `poker_hands_train.json` exists:

```bash
PYTHONPATH=. python workspace/_subnet_target/preprocess/build_dataset.py \
  --chunk-count 10000 \
  --shard-size 1000 \
  --resume \
  --preset training-merged \
  --bot-profile-mode mixed \
  --bot-candidate-attempts 3 \
  --bot-generation-rounds 1 \
  --seed 42 \
  --progress-every-chunks 100
```

Explicit paths (same as defaults here):

```bash
PYTHONPATH=. python workspace/_subnet_target/preprocess/build_dataset.py \
  --human-json workspace/_subnet_target/dataset/source/poker_hands_train.json \
  --out workspace/_subnet_target/dataset/unpreprocessed/train \
  --chunk-count 10000 \
  --shard-size 1000 \
  --resume \
  --preset training-merged \
  --bot-profile-mode mixed \
  --bot-candidate-attempts 3 \
  --bot-generation-rounds 1 \
  --seed 42 \
  --progress-every-chunks 100
```

## Shards: merge only

If shards already exist under `out/shards/`:

```bash
PYTHONPATH=. python workspace/_subnet_target/preprocess/build_dataset.py \
  --merge-only workspace/_subnet_target/dataset/unpreprocessed/train \
  --val-size 0.2 \
  --seed 42
```

## CLI / `bittensor`

Importing `hands_generator.mixed_dataset_provider` loads `bittensor`, which **extends** Python’s `argparse` globally. `--help` may list bittensor logging flags first; subnet flags (`--chunk-count`, `--human-json`, etc.) are still accepted.

## More context

- `workspace/docs/DATASET.md` — dataset notes
- Module docstring in `build_dataset.py` — miner feature schema, sharded mode, row semantics (one Parquet row = one chunk)
