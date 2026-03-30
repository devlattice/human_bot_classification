# `hands_generator` — human/bot hand JSON for Poker44

This package produces **V0 hand JSON** (`poker44/core/hand_json.py`) suitable for validators and miners after **`sanitize_hand_for_miner`**.

## Is it enough for the project?

**Technically — yes** for the subnet’s intended pipeline:

| Need | Where it lives |
|------|----------------|
| Valid synthetic bots | `bot_hands/generate_poker_data.py` → `PokerHandGenerator`, `TableSession`, `SandboxPokerBot` |
| Default bot “personalities” (single source of truth) | `bot_hands/default_bot_profiles.py` → used by `data_generator`, `generate_poker_data`, and mixed chunks in `"default"` mode |
| Wider bot behavior without a second engine | `bot_hands/extra_bot_profiles.py`, `diverse_bot_generator.py`, `mixed_dataset_provider.MixedDatasetConfig.bot_profile_mode` |
| Public human corpus | `human_hands/poker_hands_combined.json` or `.json.gz` (prefer `.gz` when present) |
| Validator-like mixed chunks + sanitization | `mixed_dataset_provider.py` |
| Quick random labeled chunks (debug) | `data_generator.py` |
| Schema sanity checks | `consistency_checker.py` |

**Logically — with clear limits:** one **rule-based** simulator + profiles covers **diverse training data** and **subnet-consistent JSON**. It does **not** emulate every external poker AI (solvers, RL bots, etc.); that is optional integration work (see `workspace/docs/BOT_DIVERSITY.md`).

## Human corpus path

`human_hands/corpus_paths.resolve_default_human_corpus_path()` (and **`MixedDatasetConfig`** defaults) resolve **`poker_hands_combined.json.gz`** first, then **`poker_hands_combined.json`**. Large files are streamed where possible. Improving **diversity and volume** of human hands helps **human FPR** — see `workspace/docs/HUMAN_CORPUS.md`.

## Imports from repo root

```bash
PYTHONPATH=. python hands_generator/data_generator.py --help
PYTHONPATH=. python hands_generator/bot_hands/generate_poker_data.py --one
```

## Related docs

- `workspace/docs/DATASET.md` — LightGBM preprocessing, train/serve parity  
- `workspace/docs/BOT_DIVERSITY.md` — profile pools vs external engines  
