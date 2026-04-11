from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

RAW_PATH       = Path("data/raw/cards_seed.csv")
PROCESSED_PATH = Path("data/processed/cards_clean.csv")
COLS           = ["name", "type", "mana_cost", "power", "toughness", "rarity", "colors"]

_COLORS      = {"W", "U", "B", "R", "G"}
_COLOR_ORDER = "WUBRG"   # canonical ordering for color_identity strings

# Priority order: first match wins for cards with multiple types (e.g. Artifact Creature -> Creature).
_BASE_TYPES = ["Creature", "Planeswalker", "Battle", "Enchantment", "Artifact", "Land", "Instant", "Sorcery"]


def parse_type_line(type_line: str | float) -> dict[str, str | None]:
    """Extract card_type and subtype from a MTG type line.

    The type line has the form: [Supertypes] CardTypes [— Subtypes]

    Examples:
        "Legendary Creature — Serpent"  → card_type="Creature",  subtype="Serpent"
        "Artifact — Equipment"          → card_type="Artifact",  subtype="Equipment"
        "Instant"                       → card_type="Instant",   subtype=None
        "Artifact Creature — Golem"     → card_type="Creature",  subtype="Golem"
    """
    if pd.isna(type_line):
        return {"card_type": "Other", "subtype": None}

    parts        = str(type_line).split("—")
    type_section = parts[0].strip()

    card_type = "Other"
    for base in _BASE_TYPES:
        if base in type_section:
            card_type = base
            break

    subtype = None
    if len(parts) > 1:
        first_word = parts[1].strip().split()
        if first_word:
            subtype = first_word[0]

    return {"card_type": card_type, "subtype": subtype}


def engineer_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'type' with 'card_type' (≤8 values) and 'subtype' (first word).

    Reduces type cardinality from ~192 unique strings to two lower-cardinality
    columns the model can correlate with color and stats.
    """
    logger.info("Engineering type features from 'type'")

    type_features = df["type"].apply(parse_type_line).apply(pd.Series)
    df = pd.concat([df.drop(columns="type"), type_features], axis=1)

    logger.debug("card_type distribution:\n{}", df["card_type"].value_counts().to_string())
    return df


def _token_cmc(token: str) -> int:
    """Converted mana cost contributed by a single mana symbol.

    Examples: "3" → 3,  "G" → 1,  "X" → 0,  "W/U" → 1
    """
    if token.isdigit():
        return int(token)
    if token in _COLORS or token == "C":
        return 1
    if token in {"X", "Y", "Z"}:
        return 0
    if "/" in token:                        # hybrid or phyrexian (e.g. W/U, W/P)
        return 1
    return 1                               # snow (S), etc.


def _token_pips(token: str) -> set[str]:
    """Color pips contributed by a single mana symbol.

    Pure colors contribute one pip. Hybrid symbols contribute one pip per
    color inside (e.g. {W/U} → W and U). Phyrexian {W/P} → W only.
    """
    if token in _COLORS:
        return {token}
    if "/" in token:
        return {part for part in token.split("/") if part in _COLORS}
    return set()


def parse_mana_cost(mana_cost: str | float) -> dict[str, int | str]:
    """Parse a MTG mana cost string into CMC and color identity.

    Returns keys: cmc (int), color_identity (str).
    color_identity is the WUBRG-sorted string of colors present in the cost,
    or "colorless" when no colored pips appear.

    Examples:
        "{2}{G}{G}"  → cmc=4, color_identity="G"
        "{W}{U}"     → cmc=2, color_identity="WU"
        "{X}{R}{R}"  → cmc=2, color_identity="R"     (X contributes 0 to cmc)
        "{W/U}"      → cmc=1, color_identity="WU"    (hybrid counts both)
        "{3}"        → cmc=3, color_identity="colorless"
    """
    if pd.isna(mana_cost):
        return {"cmc": 0, "color_identity": "colorless"}

    tokens         = re.findall(r"\{([^}]+)\}", str(mana_cost))
    colors_present = set()
    for token in tokens:
        colors_present |= _token_pips(token)

    color_identity = "".join(c for c in _COLOR_ORDER if c in colors_present) or "colorless"

    return {
        "cmc":            sum(_token_cmc(t) for t in tokens),
        "color_identity": color_identity,
    }


def engineer_mana_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'mana_cost' and 'colors' with cmc (int) and color_identity (categorical).

    'colors' is dropped because color_identity already encodes the same information
    in a single, model-friendly categorical column.
    """
    logger.info("Engineering mana features from 'mana_cost'")

    mana_features = df["mana_cost"].apply(parse_mana_cost).apply(pd.Series)
    df = pd.concat(
        [df.drop(columns=["mana_cost", "colors"], errors="ignore"), mana_features],
        axis=1,
    )

    logger.debug(
        "CMC range: {:.0f}–{:.0f}, mean: {:.1f}",
        df["cmc"].min(), df["cmc"].max(), df["cmc"].mean(),
    )
    logger.debug("color_identity distribution:\n{}", df["color_identity"].value_counts().to_string())
    return df


def engineer_stat_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add power_per_cmc and toughness_per_cmc for cards with stats.

    These ratios teach the model that a 3-mana card shouldn't have power=11.
    Cards without stats or with cmc=0 get NaN (handled by the stratified split).
    """
    has_stats = df["power"].notna() & df["toughness"].notna()
    safe_cmc  = df["cmc"].where(df["cmc"] > 0)          # NaN where cmc==0

    df["power_per_cmc"]     = pd.NA
    df["toughness_per_cmc"] = pd.NA

    df.loc[has_stats, "power_per_cmc"]     = (df.loc[has_stats, "power"]     / safe_cmc[has_stats]).round(2)
    df.loc[has_stats, "toughness_per_cmc"] = (df.loc[has_stats, "toughness"] / safe_cmc[has_stats]).round(2)

    logger.debug(
        "power_per_cmc — mean: {:.2f}, max: {:.2f}",
        df["power_per_cmc"].mean(), df["power_per_cmc"].max(),
    )
    return df


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
        after   = df[col].isna().sum()
        logger.debug("Column '{}': {:,} non-numeric value(s) coerced to NaN", col, after - before)
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.success("Saved {:,} rows to {}", len(df), path)


class PreprocessStage:
    name = "preprocess"

    def run(self, ctx: Context) -> Context:
        if ctx.raw is None:
            logger.error("No raw data in context — was the fetch stage skipped?")
            raise RuntimeError("PreprocessStage requires ctx.raw")
        df = select_columns(ctx.raw, COLS)
        df = clean(df)
        df = engineer_type_features(df)
        df = engineer_mana_features(df)
        df = engineer_stat_ratios(df)
        save(df, PROCESSED_PATH)
        ctx.clean = df
        return ctx
