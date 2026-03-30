# `workspace/utils`

Small command-line helpers for local analysis.

## `comapre_distribution_json.py`

Compares **per-hand scalar features** across two or more poker hand JSON files (useful for **domain shift** / exploratory checks). Each input must be a **top-level JSON array** of hand objects (`metadata`, `players`, `actions`, `streets`, `outcome`, optional `label`). Dict wrappers with a `hands` key are only supported when doing a **full** `json.load` on smaller files.

### Behaviour

- Derives numeric features per hand (stacks, action counts, normalized BB stats, pot summaries, etc.).
- Prints **pandas `describe()`-style** summaries and, when **SciPy** is available, **Kolmogorov–Smirnov** tables (first dataset vs each other; larger KS statistic ⇒ more distributional difference).
- Writes **CSVs** and **PNG** figures under the output directory.

### Memory and large files

To avoid loading multi-hundred-megabyte arrays into RAM:

- Files **≥ `--large-file-mb`** (default **16**) are **not** fully loaded unless you pass **`--allow-full-json-load`**.
- If **`--max-hands` is omitted** on such files, the tool streams the first **`--default-max-hands`** (default **50_000**) hands per file.
- With **`--max-hands`**, parsing streams: **`ijson`** if installed (faster), otherwise the stdlib **`JSONDecoder.raw_decode`** loop over the top-level array. Pretty-printed arrays (newlines between `}` and `,`) are supported; the streamer strips a leading comma when a read chunk starts with `,` after `}`.

### Baseline and “domain shift”

There is **no fixed external baseline** (e.g. a canonical poker corpus). The tool compares **empirical distributions of the same hand-level scalar features** across the files you pass.

- **QQ plots** (`qq_vs_first.png`) use the **first** `--input-json` as the reference quantiles; other datasets are plotted against it.
- **KS tables** (stdout): compare the **first** dataset to **each** additional one (larger KS ⇒ stronger evidence those **1D marginals** differ on the sampled hands).
- **KS CSV on disk:** only **`ks_<first_stem>__vs__<second_stem>.csv`** is written. Pairwise KS for first vs third, first vs fourth, etc. appears in the terminal only unless you save it yourself.
- **`--max-hands`** caps each file to the **first N streamed** hands (not a random draw); if the file is shorter (e.g. a small bot export), the row count is the full file length.
- Duplicate basenames get suffixed labels (`name_0`, `name_1`, …) so output filenames stay unique.

Treat many KS tests as **exploratory** (no multiplicity correction).

### Outputs (default directory: `workspace/utils/plots`)

| Artifact | Description |
|----------|-------------|
| `summary_<stem>.csv` | Per-dataset numeric summary + missing fraction |
| `ks_<A>__vs__<B>.csv` | KS statistic / p-value (**first** vs **second** path only; extra pairwise KS is stdout-only) |
| `boxplots.png` | Side-by-side boxplots (`showfliers=False`) |
| `ecdf.png` | Overlaid empirical CDFs |
| `histograms.png` | Normalized overlapping histograms |
| `qq_vs_first.png` | QQ plots vs the **first** `--input-json` dataset |

### Dependencies

- **Required:** `numpy`, `pandas` (repo base requirements).
- **Plots:** `matplotlib`.
- **KS tables:** `scipy` (often pulled in via `scikit-learn`).
- **Optional (streaming speed):** `ijson`.

### CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--input-json` | (required, repeat) | Path to each JSON dataset |
| `--out-dir` | `workspace/utils/plots` | Where to write CSVs and PNGs |
| `--max-hands` | auto for large files | Cap hands per file (streaming) |
| `--default-max-hands` | `50000` | Cap when file is large and `--max-hands` omitted |
| `--large-file-mb` | `16` | Threshold for “large file” behaviour |
| `--allow-full-json-load` | off | Force full `json.load` (OOM risk) |
| `--min-ks-n` | `30` | Min samples per side for KS |
| `--top-features` | `12` | Features shown in multi-panel figures (ranked by KS when possible) |
| `--plot-dpi` | `120` | Figure DPI |
| `--hist-bins` | `32` | Histogram bins |
| `--qq-quantiles` | `100` | QQ plot quantile count |
| `--no-plots` | off | CSV + stdout only |

### Example (two datasets)

From the repo root:

```bash
python workspace/utils/comapre_distribution_json.py \
  --input-json workspace/real_distribution/processed/merged_labeled.json \
  --input-json hands_generator/human_hands/poker_hands_combined.json \
  --max-hands 30000 \
  --out-dir workspace/utils/plots
```

### Example (four datasets)

Repeat **`--input-json`** once per file. **Order matters:** the first file is the QQ reference and the anchor for printed KS tables (first vs second, first vs third, first vs fourth). Boxplots, ECDFs, and histograms include **all** datasets on the same panels.

```bash
python workspace/utils/comapre_distribution_json.py \
  --input-json hands_generator/human_hands/poker_hands_combined.json \
  --input-json hands_generator/bot_hands/bot_hands.json \
  --input-json workspace/datasets/zenodo/zenodo_holdout.json \
  --input-json workspace/real_distribution/processed/merged_labeled.json \
  --max-hands 20000 \
  --out-dir workspace/utils/plots_four_sources
```

Use a dedicated **`--out-dir`** when comparing many sources so you do not overwrite a previous two-way run under `workspace/utils/plots`.

### Troubleshooting

**`PermissionError` when writing CSVs or PNGs**

Default output is `workspace/utils/plots`. If that directory or existing files there were created as **root** (e.g. Docker, `sudo`), your normal user cannot overwrite them.

The script exits with a short hint; you can fix it in either of these ways:

```bash
# Fix ownership of the default plots directory (from repo root)
sudo chown -R "$USER:$(id -gn)" workspace/utils/plots
```

Or write to a directory you own:

```bash
python workspace/utils/comapre_distribution_json.py \
  --input-json workspace/real_distribution/processed/merged_labeled.json \
  --input-json hands_generator/human_hands/poker_hands_combined.json \
  --out-dir /tmp/poker44_dist_plots
```
