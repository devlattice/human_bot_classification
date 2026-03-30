# GPT-readable prompts (`ssl_data/gpt`)

Tools to turn **score-split JSON** (e.g. `split_out/score_medium.json`) into **small text prompts** for an LLM (OpenAI or similar). This avoids sending multi-megabyte raw `chunk` JSON per request while keeping enough structure (counts + optional action shorthand) for a binary **human vs bot** style label.

Training rows for the miner pipeline still come from **`aggregate_chunk_from_hands` + `transform_meta`**: use these prompts only for **labeling**; merge predictions back onto the original records as **`label` + `chunk`** for `build_dataset.py --refine-score-json`. The optional script **`openai_label.py`** calls the OpenAI API for each prompt file, writes that refine JSON, and can stamp **`gpt_score`** back onto the full source array. (It must not be named `openai.py`, or `import openai` would load this file instead of the PyPI package.)

## Layout

| File | Role |
|------|------|
| `make_gpt_readable.py` | CLI: JSON array → `prompts/prompt_NNNNNN.txt` + stdout token/cost estimates. |
| `openai_label.py` | CLI: `prompt_*.txt` + `--source` → `refine-score-json.json` + augmented source JSON with **`gpt_score`**. |
| `system_prompt_stub.txt` | Example **system** message (rubric + JSON-only reply). Used for **token/cost** estimate when the file exists; also the default system message for **`openai_label.py`**. |
| `example_compact_user_message.txt` | Reference **user** message shape (matches generator defaults). |
| `size_estimate_vs_full_json.txt` | Rough size ratio vs pasting full `score_medium.json`. |

## Input format

A **JSON array** of objects, same family as `split_by_score` outputs:

- **`chunk`** (required): list of hand dicts (miner-visible schema: `metadata`, `players`, `actions`, …).
- **`chunk_hash`** (optional): written into the prompt as `chunk_hash: <hex>` for joining; if missing → `chunk_hash: (none)`.
- **`risk_score`** (optional): teacher score, one line in the prompt (not a ground-truth label).

Non-dict array entries are skipped. Empty `chunk` lists are skipped (counted on stderr).

## Output layout

With `--output-dir DIR`:

- **`DIR/prompts/prompt_000000.txt`**, **`prompt_000001.txt`**, … — one file per **array index** in the input (gaps only if that index was skipped).
- Each file contains: `chunk_hash` line, chunk-level **action/street counts**, optional **risk_score** line, **ACTION SHORTHAND** for the first *N* valid hand dicts (`--max-shorthand-hands`), then the **TASK** line.

**Joining:** use **`chunk_hash`** from the prompt text, or **`prompt_XXXXXX.txt`** ↔ input array index **`XXXXXX`**, to attach the model’s `label` to the matching source object’s **`chunk`** when building `refine-score-json.json`.

## CLI (`make_gpt_readable.py`)

Run from repo root (`miner_1`) so default `--input-json` resolves:

```bash
cd /path/to/miner_1

python3 workspace/datasets/ssl_data/gpt/make_gpt_readable.py \
  --input-json workspace/datasets/ssl_data/split_out/score_medium.json \
  --output-dir workspace/datasets/ssl_data/gpt/prompts_out
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input-json` | `workspace/datasets/ssl_data/split_out/score_medium.json` | Source JSON array. |
| `--output-dir` | *(required)* | Parent directory; writes under `prompts/`. |
| `--max-shorthand-hands` | `1` | How many hands get a shorthand block (more tokens). |
| `--include-outlier-placeholder` | off | Adds optional z-score template section. |
| `--system-prompt-file` | `system_prompt_stub.txt` next to script | Used only for **token/cost** estimate. |
| `--usd-per-1m-input` / `--usd-per-1m-output` | `0.15` / `0.60` | Rough **GPT-4o mini** scale; override for other models. |
| `--assumed-output-tokens` | `12` | Per-call completion length for cost line. |

**Stderr:** progress (`records_written`, `skipped_empty`, output path). **Stdout:** `token_backend`, averages, totals, `estimated_cost_USD`.

**Token counting:** uses **`tiktoken`** (`gpt-4o` encoding) when installed; otherwise **`len(text)//4`**. Install `tiktoken` for more accurate estimates.

**Large files:** if **`ijson`** is installed, streaming parse is tried first; on failure the script falls back to **`json.load`**.

## OpenAI labeling (`openai_label.py`)

Calls **one chat completion per** `prompt_NNNNNN.txt` (sequential, CPU- and rate-limit friendly). Parses the model reply as JSON `{"label":0}` or `{"label":1}` (same convention as `system_prompt_stub.txt`).

**Dependencies:** `pip install openai`. **`ijson`** is optional but recommended for large `--source` files when only a subset of indices is labeled (lighter than loading the whole array just to read a few rows).

**Environment:** `OPENAI_API_KEY` must be set.

**Outputs**

| Output | Description |
|--------|-------------|
| `DIR/refine-score-json.json` | JSON array for `build_dataset.py --refine-score-json`: each object has **`label`**, **`chunk`**, and copies **`chunk_hash`** / **`risk_score`** when present on the source row. `DIR` is `--output` (default `gpt/prompts_out/refined`). |
| Augmented source JSON | Full **`--source`** array rewritten with an extra integer field **`gpt_score`** (`0` or `1`) on every row that was successfully labeled. Rows that were skipped or failed are left unchanged (no `gpt_score` added). Default destination: **`<source_stem>_gptscore.json`** beside `--source`. Use **`--in-place`** to overwrite `--source` instead. |

**Stderr:** progress lines prefixed with `[gpt-label]` (flushed).

**Example**

```bash
cd /path/to/miner_1
export OPENAI_API_KEY=...

python3 workspace/datasets/ssl_data/gpt/openai_label.py \
  --input-data workspace/datasets/ssl_data/gpt/prompts_out \
  --source workspace/datasets/ssl_data/split_out/score_medium.json \
  --output workspace/datasets/ssl_data/gpt/prompts_out/refined
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input-data`, `--input-dta` | `workspace/datasets/ssl_data/gpt/prompts_out` | Directory with `prompts/` or top-level `prompt_*.txt`. |
| `--source` | `split_out/score_medium.json` (under workspace defaults) | JSON array aligned by prompt index. |
| `--output`, `--out-put`, `--out_put` | `prompts_out/refined` | Directory for **`refine-score-json.json`**. |
| `--source-out` | *(see above)* | Path for augmented source with **`gpt_score`**. Ignored if **`--in-place`**. |
| `--in-place` | off | Write augmented JSON over **`--source`**. |
| `--system-prompt-file` | `system_prompt_stub.txt` | System message. |
| `--model` | `gpt-4o-mini` | Chat completion model. |
| `--sleep-seconds` | `0` | Pause after each successful request. |
| `--timeout` | `120` | HTTP timeout per request (seconds). |
| `--max-retries` | `3` | Retries per prompt on failure. |
| `--limit` | `0` | Process at most *N* prompts (`0` = all). |
| `--dry-run` | off | No API calls; no files written; logs planned **`--source-out`** path. |

**Exit codes:** `0` if no API failures; `2` if any prompt failed after retries; `1` for bad paths or no prompts.

## Downstream (`build_dataset.py`)

Refined data for training must be a JSON array of objects with:

- **`label`**: `0` or `1` (required).
- **`chunk`**: same hand list as in the source JSON (required).

Do **not** pass pre-robust-scaled feature rows as `chunk`; the builder runs **`aggregate_chunk_from_hands`** and **`transform_meta`** again.

```bash
PYTHONPATH=. python3 workspace/datasets/ssl_data/build_dataset.py \
  --low-score-json ... \
  --high-score-json ... \
  --refine-score-json path/to/refine-score-json.json \
  ...
```

See `workspace/datasets/ssl_data/readme.md` for the full SSL pipeline.
