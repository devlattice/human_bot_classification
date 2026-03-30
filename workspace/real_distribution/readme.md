# Real-distribution hands

Utilities and processed dumps that align **real** hand histories with the same JSON schema as the shipped human corpus (`hands_generator/human_hands/poker_hands_combined.json`), i.e. Poker44 **V0** hand objects (`poker44.core.hand_json.V0_JSON_HAND`).

## Layout

| Path | Notes |
|------|--------|
| `processed/merged.json` | Typically **NDJSON** (one JSON object per line), despite the `.json` name. Same nested fields as V0 hands but usually **no** top-level `label`. |
| `processed/merged.jsonl` | Same line-oriented format as above, if you keep a `.jsonl` copy. |
| `transfer.py` | Normalizes to V0 key order, adds `label`, deduplicates, writes JSON array or NDJSON. |

## `transfer.py`

Run from the **repository root** so `poker44` is importable:

```bash
PYTHONPATH=. python workspace/real_distribution/transfer.py --help
```

### Defaults

- **Input:** `workspace/real_distribution/processed/merged.json`
- **Output:** `workspace/real_distribution/processed/merged_labeled.json`
- **Output format:** JSON **array** (`[ {...}, {...} ]`), compact (no indent), same container style as `poker_hands_combined.json`.

### Behavior

1. **Schema** — Output objects have top-level keys in V0 order: `metadata`, `players`, `streets`, `actions`, `outcome`, `label`. Any **extra** top-level keys in the input cause an error.
2. **Labels** — If `label` is missing, it is set to JSON **`null`** (pandas reads this as NaN). If `label` is already present (e.g. `"human"`, `"bot"`), it is kept. Optional: `--unknown-nan` uses float NaN in Python output (non-standard JSON unless `--allow-json-nan` / implied when using `--unknown-nan`).
3. **Deduplication (on by default)** — A duplicate is the same hand **history**: SHA-256 of canonical JSON for `metadata`, `players`, `streets`, `actions`, `outcome` only (`label` is ignored). **First row wins.** Logs `input`, `unique`, and `duplicates_dropped`. Use `--no-deduplicate` to emit every row.
4. **Input shape** — If the file’s first non-whitespace character is `[`, the whole file is read as one JSON **array**. Otherwise each non-empty line is one JSON **object** (NDJSON).

### Examples

```bash
# Defaults: merged.json → merged_labeled.json (array, deduped, label null if missing)
PYTHONPATH=. python workspace/real_distribution/transfer.py

# Explicit paths
PYTHONPATH=. python workspace/real_distribution/transfer.py \
  --input workspace/real_distribution/processed/merged.json \
  --output workspace/real_distribution/processed/merged_labeled.json

# NDJSON output (one hand per line)
PYTHONPATH=. python workspace/real_distribution/transfer.py \
  -i workspace/real_distribution/processed/merged.json \
  -o workspace/real_distribution/processed/merged_labeled.jsonl \
  --format ndjson

# Pretty-printed array (large files)
PYTHONPATH=. python workspace/real_distribution/transfer.py --indent 2

# No deduplication
PYTHONPATH=. python workspace/real_distribution/transfer.py --no-deduplicate
```

### Training note

`label: null` marks **unknown** supervision; combine this file with fully labeled corpora only if your training code explicitly handles missing labels (e.g. filter, semi-supervised loss, or separate splits). Validator/miner-facing hand views may still strip labels via sanitization—train on the same view you will see at inference when applicable.
