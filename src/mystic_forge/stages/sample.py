from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

MODEL_STATS    = Path("output/synth_with_stats.pkl")
MODEL_NO_STATS = Path("output/synth_no_stats.pkl")
SYNTHETIC_PATH = Path("data/synthetic/cards_synthetic.csv")
DEFAULT_ROWS   = 300


def load_model(path: Path) -> CTGANSynthesizer:
    if not path.exists():
        logger.error("Model file not found: {}", path)
        raise FileNotFoundError(path)
    logger.info("Loading model from {}", path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CTGANSynthesizer.load(str(path))
    return model


def sample(model: CTGANSynthesizer, num_rows: int, label: str) -> pd.DataFrame:
    logger.info("Sampling {:,} '{}' rows", num_rows, label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = model.sample(num_rows=num_rows)
    logger.debug("'{}' shape: {} rows x {} columns", label, *df.shape)
    return df


def reconstruct_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct integer power/toughness from per-cmc ratios, then drop the ratios.

    The stats model is trained on power_per_cmc and toughness_per_cmc instead of
    absolute values so that CTGAN learns a distribution that stays proportional to
    cmc. This function inverts the transform:

        power     = round(power_per_cmc     × max(cmc, 1))
        toughness = round(toughness_per_cmc × max(cmc, 1))

    cmc is clipped to a minimum of 1 to handle 0-cost cards (e.g. Ornithopter)
    whose ratio would be NaN in training data. A NaN ratio is filled with 1.0
    (one point of power/toughness per effective mana) as a safe default.
    """
    effective_cmc = df["cmc"].clip(lower=1)
    df["power"]     = (df["power_per_cmc"].fillna(1.0)     * effective_cmc).round().clip(lower=0).astype(int)
    df["toughness"] = (df["toughness_per_cmc"].fillna(1.0) * effective_cmc).round().clip(lower=0).astype(int)
    return df.drop(columns=["power_per_cmc", "toughness_per_cmc"])


def merge(df_stats: pd.DataFrame, df_no_stats: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df_stats, df_no_stats], ignore_index=True).sample(frac=1).reset_index(drop=True)
    logger.info("Merged {:,} synthetic rows ({:,} with stats, {:,} without)",
                len(df), len(df_stats), len(df_no_stats))
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.success("Saved {:,} synthetic rows to {}", len(df), path)


class SampleStage:
    name = "sample"

    def run(self, ctx: Context) -> Context:
        path_stats    = Path(ctx.meta.get("model_stats", str(MODEL_STATS)))
        path_no_stats = Path(ctx.meta.get("model_no_stats", str(MODEL_NO_STATS)))
        ratio         = ctx.meta.get("stats_ratio", 0.5)

        n_stats    = math.ceil(DEFAULT_ROWS * ratio)
        n_no_stats = DEFAULT_ROWS - n_stats
        logger.info("Sampling ratio: {:.1%} with stats, {:.1%} without", ratio, 1 - ratio)

        model_stats    = load_model(path_stats)
        model_no_stats = load_model(path_no_stats)

        df_stats    = sample(model_stats, n_stats, "with_stats")
        df_no_stats = sample(model_no_stats, n_no_stats, "no_stats")

        df_stats = reconstruct_stats(df_stats)
        df = merge(df_stats, df_no_stats)
        save(df, SYNTHETIC_PATH)
        return ctx
