from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

CLEAN_PATH     = Path("data/processed/cards_clean.csv")
OUTPUT_DIR     = Path("output")
MODEL_STATS    = OUTPUT_DIR / "synth_with_stats.pkl"
MODEL_NO_STATS = OUTPUT_DIR / "synth_no_stats.pkl"

EXCLUDE_COLS = ["name"]
EPOCHS       = 1000

# Columns only meaningful for cards that have stats (power/toughness).
STATS_COLS = ["power", "toughness", "power_per_cmc", "toughness_per_cmc"]


def load_clean(path: Path) -> pd.DataFrame:
    logger.info("Reading clean data from {}", path)
    df = pd.read_csv(path)
    logger.info("Loaded {:,} rows, {:,} columns", len(df), len(df.columns))
    return df


def split_by_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["power"].notna() & df["toughness"].notna()
    has = df[mask]
    no  = df[~mask].drop(columns=STATS_COLS)
    logger.info("Stratified split: {:,} with stats, {:,} without", len(has), len(no))
    return has, no


def build_metadata(df: pd.DataFrame, categorical: list[str]) -> Metadata:
    logger.debug("Detecting metadata from DataFrame")
    metadata = Metadata.detect_from_dataframe(df, table_name="cards")
    for col in categorical:
        if col in df.columns:
            metadata.update_column(col, table_name="cards", sdtype="categorical")
    logger.debug("Categorical columns set: {}", categorical)
    return metadata


def fit_model(df: pd.DataFrame, metadata: Metadata, label: str) -> CTGANSynthesizer:
    logger.info("Training '{}' model on {:,} rows ({} epochs)", label, len(df), EPOCHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CTGANSynthesizer(metadata, epochs=EPOCHS, verbose=True)
        model.fit(df)
    logger.info("Training '{}' complete", label)
    return model


def save_model(model: CTGANSynthesizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.success("Saved model to {}", path)


class TrainStage:
    name = "train"

    def run(self, ctx: Context) -> Context:
        if ctx.clean is None:
            logger.error("No clean data in context — was the preprocess stage skipped?")
            raise RuntimeError("TrainStage requires ctx.clean")

        df = ctx.clean.drop(columns=[c for c in EXCLUDE_COLS if c in ctx.clean.columns])
        logger.debug("Excluded columns before training: {}", EXCLUDE_COLS)

        df_stats, df_no_stats = split_by_stats(df)
        ratio = len(df_stats) / len(df)

        cat_stats    = ["rarity", "card_type", "subtype", "color_identity"]
        cat_no_stats = ["rarity", "card_type", "subtype", "color_identity"]

        # Train the stats model on ratios (power_per_cmc, toughness_per_cmc) rather
        # than absolute values. This lets CTGAN learn a compact distribution that
        # stays consistent with cmc. Power and toughness are reconstructed at
        # sample time: power = round(power_per_cmc × cmc).
        df_stats_train = df_stats.drop(columns=["power", "toughness"])

        meta_stats    = build_metadata(df_stats_train, cat_stats)
        meta_no_stats = build_metadata(df_no_stats, cat_no_stats)

        model_stats    = fit_model(df_stats_train, meta_stats, "with_stats")
        model_no_stats = fit_model(df_no_stats, meta_no_stats, "no_stats")

        save_model(model_stats, MODEL_STATS)
        save_model(model_no_stats, MODEL_NO_STATS)

        ctx.meta["model_stats"]    = str(MODEL_STATS)
        ctx.meta["model_no_stats"] = str(MODEL_NO_STATS)
        ctx.meta["stats_ratio"]    = ratio

        return ctx