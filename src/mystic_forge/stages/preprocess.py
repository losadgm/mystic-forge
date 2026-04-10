from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

RAW_PATH = Path("data/raw/cards_seed.csv")
PROCESSED_PATH = Path("data/processed/cards_clean.csv")
COLS = ["name", "type", "mana_cost", "power", "toughness", "rarity", "colors"]


def load_raw(path: Path) -> pd.DataFrame:
    logger.info("Reading raw data from {}", path)
    df = pd.read_csv(path)
    logger.info("Loaded {:,} rows, {:,} columns", len(df), len(df.columns))
    return df


def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("Columns not found in source, will be skipped: {}", missing)
        cols = [c for c in cols if c in df.columns]
    df = df[cols]
    logger.debug("Selected columns: {}", list(df.columns))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("power", "toughness"):
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        coerced = after - before
        logger.debug(
            "Column '{}': {:,} non-numeric value(s) coerced to NaN", col, coerced
        )

    if "colors" in df.columns:
        filled = df["colors"].isna().sum()
        df["colors"] = df["colors"].fillna("C")
        logger.debug("Column 'colors': {:,} missing value(s) filled with 'C'", filled)

    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.success("Saved {:,} rows to {}", len(df), path)


class PreprocessStage:
    name = "preprocess"

    def run(self, ctx: Context) -> Context:
        df = select_columns(ctx.raw, COLS)
        df = clean(df)
        save(df, PROCESSED_PATH)
        ctx.clean = df
        return ctx
