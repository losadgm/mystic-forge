from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

CLEAN_PATH     = Path("data/processed/cards_clean.csv")
OUTPUT_DIR     = Path("output")
MODEL_STATS    = OUTPUT_DIR / "synth_with_stats.pkl"
MODEL_NO_STATS = OUTPUT_DIR / "synth_no_stats.pkl"

# cmc is a deterministic function of pip_* and generic; excluding it prevents the
# model from treating a redundant derived column as an independent variable, which
# would cause unavoidable drift at sample time. It is recomputed from pips in
# enforce_mana_invariants (validate stage).
EXCLUDE_COLS       = ["name", "cmc"]
EPOCHS             = 2000
_MIN_TRAINING_ROWS = 20   # below this CTGAN will produce near-random output

# Columns only meaningful for cards that have stats (power/toughness).
STATS_COLS = ["power", "toughness", "power_per_cmc", "toughness_per_cmc"]


def load_clean(path: Path) -> pd.DataFrame:
    logger.debug("Reading clean data from {}", path)
    df = pd.read_csv(path)
    logger.debug("Loaded {:,} rows, {:,} columns", len(df), len(df.columns))
    return df


def split_by_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["power"].notna() & df["toughness"].notna()
    has = df[mask]
    no  = df[~mask].drop(columns=STATS_COLS)
    logger.info("Stratified split: {:,} with stats, {:,} without", len(has), len(no))
    if len(has) < _MIN_TRAINING_ROWS:
        logger.warning(
            "Only {:,} rows in the 'with stats' subset (minimum recommended: {:,}) — "
            "CTGAN may produce low-quality samples",
            len(has), _MIN_TRAINING_ROWS,
        )
    if len(no) < _MIN_TRAINING_ROWS:
        logger.warning(
            "Only {:,} rows in the 'without stats' subset (minimum recommended: {:,}) — "
            "CTGAN may produce low-quality samples",
            len(no), _MIN_TRAINING_ROWS,
        )
    return has, no


def build_metadata(df: pd.DataFrame, categorical: list[str]) -> Metadata:
    logger.debug("Building metadata from {:,} rows, {:,} columns", *df.shape)
    metadata = Metadata.detect_from_dataframe(df, table_name="cards")
    missing  = [col for col in categorical if col not in df.columns]
    if missing:
        logger.warning(
            "Categorical column(s) not found in training data and will be skipped: {}",
            missing,
        )
    for col in categorical:
        if col in df.columns:
            metadata.update_column(col, table_name="cards", sdtype="categorical")
    present = [c for c in categorical if c not in missing]
    logger.debug("Categorical columns registered: {}", present)
    logger.debug("All columns in training set: {}", list(df.columns))
    return metadata


def fit_model(df: pd.DataFrame, metadata: Metadata, label: str) -> CopulaGANSynthesizer:
    # CopulaGANSynthesizer uses CTGAN for per-column marginals and adds a Gaussian
    # copula to model cross-column correlations (e.g. cmc ↔ power_per_cmc,
    # rarity ↔ stats). This is a direct improvement over plain CTGAN for datasets
    # where inter-column relationships are as important as marginal distributions.
    logger.info("Training '{}' model on {:,} rows ({} epochs)", label, len(df), EPOCHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CopulaGANSynthesizer(metadata, epochs=EPOCHS, verbose=True)
        model.fit(df)
    logger.info("Training '{}' complete", label)
    return model


def save_model(model: CopulaGANSynthesizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.success("Saved model to {}", path)


class TrainStage:
    name = "train"

    def run(self, ctx: Context) -> Context:
        if ctx.clean is None:
            logger.error("No clean data in context — was the preprocess stage skipped?")
            raise RuntimeError("TrainStage requires ctx.clean")

        dropped = [c for c in EXCLUDE_COLS if c in ctx.clean.columns]
        df = ctx.clean.drop(columns=dropped)
        logger.debug("Dropping non-training columns: {}", dropped)

        df_stats, df_no_stats = split_by_stats(df)
        ratio = len(df_stats) / len(df)
        logger.debug("Stats/no-stats ratio: {:.1%} / {:.1%}", ratio, 1 - ratio)

        cat_cols = ["rarity", "card_type", "subtype", "color"]

        # Train the stats model on ratios (power_per_cmc, toughness_per_cmc) rather
        # than absolute values. This lets CTGAN learn a compact distribution that
        # stays consistent with cmc. Power and toughness are reconstructed at
        # sample time: power = round(power_per_cmc * cmc).
        df_stats_train = df_stats.drop(columns=["power", "toughness"])
        logger.debug("Stats model training columns: {}", list(df_stats_train.columns))
        logger.debug("No-stats model training columns: {}", list(df_no_stats.columns))

        meta_stats    = build_metadata(df_stats_train, cat_cols)
        meta_no_stats = build_metadata(df_no_stats,    cat_cols)

        model_stats    = fit_model(df_stats_train, meta_stats,    "with_stats")
        model_no_stats = fit_model(df_no_stats,    meta_no_stats, "no_stats")

        save_model(model_stats,    MODEL_STATS)
        save_model(model_no_stats, MODEL_NO_STATS)

        ctx.meta["model_stats"]    = str(MODEL_STATS)
        ctx.meta["model_no_stats"] = str(MODEL_NO_STATS)
        ctx.meta["stats_ratio"]    = ratio

        logger.info("Training complete — models saved to {}", OUTPUT_DIR)
        return ctx