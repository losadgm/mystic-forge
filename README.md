# mystic_forge

Synthetic trading card data generation experiments in Python.

Fetches real Magic: The Gathering cards from [mtgjson.com](https://mtgjson.com), engineers structured features, trains a [CTGAN](https://sdv.dev) model, and samples new synthetic cards that mimic the original distribution.

## Pipeline

The pipeline runs four stages in sequence:

```
fetch → preprocess → train → sample
```

| Stage | What it does |
|---|---|
| **fetch** | Downloads card data from mtgjson.com for a configured list of sets. Deduplicates by card name and saves to `data/raw/cards_seed.csv`. |
| **preprocess** | Selects relevant columns, parses the type line into `card_type` + `subtype`, parses mana cost into `cmc` + `color_identity`, engineers `power_per_cmc` and `toughness_per_cmc` ratios, and saves to `data/processed/cards_clean.csv`. |
| **train** | Splits cards into two subsets (with/without power & toughness) and trains one CTGAN model per subset. The stats model learns ratio-based distributions so power/toughness stay consistent with CMC. Models are saved to `output/`. |
| **sample** | Loads both models, samples rows proportional to the original stats/no-stats split, reconstructs integer power/toughness from ratios, merges and shuffles, then saves to `data/synthetic/cards_synthetic.csv`. |

## Usage

```bash
# Install dependencies (requires Python ≥ 3.13 and uv)
uv sync

# Run the full pipeline
uv run mystic-forge
```

## Project layout

```
src/mystic_forge/
    pipeline.py          # Context dataclass + Stage protocol + Pipeline runner
    run.py               # Entry point (mystic-forge CLI command)
    stages/
        fetch.py         # FetchStage — mtgjson download
        preprocess.py    # PreprocessStage — feature engineering
        train.py         # TrainStage — CTGAN training
        sample.py        # SampleStage — synthetic sampling

data/
    raw/                 # cards_seed.csv (fetched)
    processed/           # cards_clean.csv (engineered features)
    synthetic/           # cards_synthetic.csv (model output)

output/
    synth_with_stats.pkl
    synth_no_stats.pkl
```

## Sets included

The fetch stage currently pulls from four 2024 Standard-legal sets:

- **FDN** — Foundations
- **MKM** — Murders at Karlov Manor
- **BLB** — Bloomburrow
- **DSK** — Duskmourn

Add or remove entries in `stages/fetch.py` → `SETS` to change the corpus.

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and feature engineering |
| `sdv` | CTGAN synthesizer and metadata handling |
| `requests` | HTTP calls to mtgjson.com |
| `loguru` | Structured logging across all stages |
