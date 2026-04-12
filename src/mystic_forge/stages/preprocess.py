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
_COLOR_ORDER = "WUBRG"

# Priority order: first match wins for cards with multiple types (e.g. Artifact Creature -> Creature).
_BASE_TYPES = ["Creature", "Planeswalker", "Battle", "Enchantment", "Artifact", "Land", "Instant", "Sorcery"]

# Returned when mana_cost is missing (lands, tokens, etc.)
_NULL_MANA_FEATURES: dict[str, int] = {
    "cmc":     0,
    "pip_W":   0,
    "pip_U":   0,
    "pip_B":   0,
    "pip_R":   0,
    "pip_G":   0,
    "generic": 0,
}

# power or toughness per CMC above this is suspicious
_MAX_EXPECTED_RATIO = 10.0

# Subtypes with fewer than this many occurrences are collapsed into "Other".
# Raising the value reduces cardinality further; lowering it preserves more granularity.
_SUBTYPE_MIN_COUNT = 5

# Sentinel values introduced by preprocessing that should never be bucketed.
_SUBTYPE_PROTECTED = frozenset({"none", "Other"})


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


def parse_mana_cost(mana_cost: str | float) -> dict[str, int]:
    """Parse a MTG mana cost string into structured mana features.

    Returns:
        cmc           - total converted mana cost
        pip_W/U/B/R/G - count of each pure single-color symbol (e.g. {W}, {G})
        generic       - total generic mana: numeric symbols + hybrid/phyrexian + {C}

    Invariant: cmc == pip_W + pip_U + pip_B + pip_R + pip_G + generic

    color is NOT returned here — it is derived deterministically from the pip
    columns at sample time and is therefore excluded from the clean dataset.

    Hybrid symbols (e.g. {W/U}) and phyrexian symbols (e.g. {W/P}) contribute
    1 to cmc and are counted in *generic* rather than any single pip counter.
    This preserves the invariant and keeps pip columns as pure per-color counts.

    Examples:
        "{2}{G}{G}"  → cmc=4, pip_G=2, generic=2
        "{W}{U}"     → cmc=2, pip_W=1, pip_U=1, generic=0
        "{3}"        → cmc=3, (all pips=0), generic=3
    """
    if pd.isna(mana_cost):
        return _NULL_MANA_FEATURES.copy()

    tokens     = re.findall(r"\{([^}]+)\}", str(mana_cost))
    pip_counts = {c: 0 for c in _COLOR_ORDER}
    generic    = 0

    for token in tokens:
        if token.isdigit():
            generic += int(token)
        elif token in _COLORS:
            pip_counts[token] += 1
        elif token == "C":
            generic += 1
        elif "/" in token:
            # Hybrid (e.g. {W/U}) or phyrexian (e.g. {W/P}): 1 cmc, counted as generic.
            # Cards with hybrid costs are excluded in preprocess, but handled defensively.
            generic += 1
        # X, Y, Z contribute 0 to cmc — cards with {X} are excluded in preprocess

    cmc = generic + sum(pip_counts.values())

    return {
        "cmc":     cmc,
        "pip_W":   pip_counts["W"],
        "pip_U":   pip_counts["U"],
        "pip_B":   pip_counts["B"],
        "pip_R":   pip_counts["R"],
        "pip_G":   pip_counts["G"],
        "generic": generic,
    }


def engineer_mana_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'mana_cost' and 'colors' with structured mana feature columns.

    New columns: cmc (int), color (categorical), pip_W/U/B/R/G (int each), generic (int).

    color is derived from the pip columns so it is always consistent with them.
    It is included as a training feature because CopulaGAN learns categorical-categorical
    correlations (color ↔ subtype) much more reliably than numeric-categorical ones
    (pip_U ↔ subtype). In the validate stage color is re-derived from the sampled pips
    to enforce consistency in the final output.

    See parse_mana_cost for full semantics.
    """
    logger.info("Engineering mana features from 'mana_cost'")

    pip_cols = ["pip_W", "pip_U", "pip_B", "pip_R", "pip_G"]

    mana_features = df["mana_cost"].apply(parse_mana_cost).apply(pd.Series)
    df = pd.concat(
        [df.drop(columns=["mana_cost", "colors"], errors="ignore"), mana_features],
        axis=1,
    )

    # Derive color from pip columns — always consistent with pips by construction.
    df["color"] = df[pip_cols].apply(
        lambda row: "".join(sym for col, sym in zip(pip_cols, _COLOR_ORDER) if row[col] > 0)
                    or "colorless",
        axis=1,
    )

    logger.debug(
        "CMC — range: {:.0f}-{:.0f}, mean: {:.1f}",
        df["cmc"].min(), df["cmc"].max(), df["cmc"].mean(),
    )
    logger.debug("color distribution:\n{}", df["color"].value_counts().to_string())
    logger.debug(
        "Mean pip counts per card — W:{:.2f} U:{:.2f} B:{:.2f} R:{:.2f} G:{:.2f} generic:{:.2f}",
        df["pip_W"].mean(), df["pip_U"].mean(), df["pip_B"].mean(),
        df["pip_R"].mean(), df["pip_G"].mean(), df["generic"].mean(),
    )
    return df


def engineer_stat_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add power_per_cmc and toughness_per_cmc for cards with stats.

    These ratios teach the model that a 3-mana card shouldn't have power=11.
    Cards without stats or with cmc=0 get NaN (handled by the stratified split).
    """
    has_stats  = df["power"].notna() & df["toughness"].notna()
    n_stats    = int(has_stats.sum())
    n_no_stats = len(df) - n_stats
    logger.info("Stat ratios: {:,} cards with stats, {:,} without", n_stats, n_no_stats)

    if n_stats == 0:
        logger.warning(
            "No cards with power/toughness found — stats model will have no training data"
        )

    safe_cmc = df["cmc"].where(df["cmc"] > 0)   # NaN where cmc==0

    df["power_per_cmc"]     = pd.NA
    df["toughness_per_cmc"] = pd.NA

    df.loc[has_stats, "power_per_cmc"]     = (df.loc[has_stats, "power"]     / safe_cmc[has_stats]).round(2)
    df.loc[has_stats, "toughness_per_cmc"] = (df.loc[has_stats, "toughness"] / safe_cmc[has_stats]).round(2)

    max_power_ratio     = df["power_per_cmc"].max()
    max_toughness_ratio = df["toughness_per_cmc"].max()
    logger.debug(
        "power_per_cmc — mean: {:.2f}, max: {:.2f}",
        df["power_per_cmc"].mean(), max_power_ratio,
    )
    if max_power_ratio > _MAX_EXPECTED_RATIO:
        logger.warning(
            "power_per_cmc max is {:.1f} — unusually high ratio may skew model distribution",
            max_power_ratio,
        )
    if max_toughness_ratio > _MAX_EXPECTED_RATIO:
        logger.warning(
            "toughness_per_cmc max is {:.1f} — unusually high ratio may skew model distribution",
            max_toughness_ratio,
        )

    return df


def exclude_non_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any card where power or toughness is present but not a valid integer.

    Must be called BEFORE clean() while power/toughness are still raw strings.
    This is essential because clean() coerces non-numeric values (e.g. '*') to NaN,
    making them indistinguishable from legitimately absent stats (lands, instants).
    Checking the raw strings is the only reliable way to distinguish:

        power=None  → absent (land/instant) → keep
        power='*'   → present but non-integer → exclude
        power='3'   → valid integer → keep

    Both the asymmetric case (power='*', toughness='4') and the symmetric case
    (power='*', toughness='*') are caught.
    """
    logger.info("Filter: removing cards with non-integer power or toughness")

    def _has_non_numeric(series: pd.Series) -> pd.Series:
        """True where the value is present but cannot be parsed as a number."""
        return series.notna() & pd.to_numeric(series, errors="coerce").isna()

    bad = _has_non_numeric(df["power"]) | _has_non_numeric(df["toughness"])
    excluded = int(bad.sum())

    remaining = len(df) - excluded
    logger.info("  → {:,} excluded (non-integer stat value, e.g. '*'), {:,} remain", excluded, remaining)
    if excluded:
        logger.debug("Excluded cards: {}", df.loc[bad, "name"].tolist())

    return df[~bad].reset_index(drop=True)


def exclude_x_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cards whose mana cost contains {X}.

    X-cost cards have a variable component that depends on the amount of mana
    spent at cast time, which cannot be represented by a fixed numeric feature.
    Excluding them prevents the model from learning a degenerate distribution
    for cmc (which would always be 0 for X cards, regardless of intent).
    """
    logger.info("Filter: removing cards with {{X}} in mana cost")
    mask      = df["mana_cost"].str.contains(r"\{X\}", na=False)
    excluded  = int(mask.sum())
    remaining = len(df) - excluded
    logger.info("  → {:,} removed, {:,} remain", excluded, remaining)
    return df[~mask].reset_index(drop=True)


def exclude_hybrid_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cards whose mana cost contains hybrid or phyrexian symbols (e.g. {W/U}, {W/P}).

    Hybrid and phyrexian symbols cannot be cleanly decomposed into pure pip counts
    and generic mana: a {W/U} symbol is counted as generic but still influences
    the color field, creating an inconsistency between the pip columns and color.
    Excluding these cards keeps the mana feature space fully consistent.
    """
    logger.info("Filter: removing cards with hybrid/phyrexian mana costs")
    mask      = df["mana_cost"].str.contains(r"\{[^}]*/[^}]*\}", na=False)
    excluded  = int(mask.sum())
    remaining = len(df) - excluded
    if excluded:
        logger.debug("Hybrid/phyrexian cards: {}", df.loc[mask, "name"].tolist())
    logger.info("  → {:,} removed, {:,} remain", excluded, remaining)
    return df[~mask].reset_index(drop=True)


def fill_null_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace null subtype with the sentinel value 'none'.

    CTGAN cannot handle NaN in categorical columns. Cards without a subtype
    (instants, sorceries, some artifacts) get a consistent 'none' label so
    the model learns it as a valid, frequent category rather than missing data.
    """
    null_count = int(df["subtype"].isna().sum())
    logger.debug(
        "Filling {:,} null subtype(s) → 'none' ({:,} cards already have a subtype)",
        null_count, int(df["subtype"].notna().sum()),
    )
    df["subtype"] = df["subtype"].fillna("none")
    logger.debug(
        "Subtype cardinality after fill: {:,} unique values",
        df["subtype"].nunique(),
    )
    return df


def bucket_rare_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse low-frequency subtypes into 'Other'.

    Subtypes with fewer than _SUBTYPE_MIN_COUNT occurrences don't provide enough
    examples for CTGAN to learn a meaningful conditional distribution. Collapsing
    them into 'Other' reduces cardinality while preserving the signal for common
    subtypes. Sentinel values ('none', 'Other') are never eligible for bucketing.
    """
    counts     = df["subtype"].value_counts()
    rare_mask  = (counts < _SUBTYPE_MIN_COUNT) & (~counts.index.isin(_SUBTYPE_PROTECTED))
    rare_types = counts[rare_mask].index

    n_types_before = df["subtype"].nunique()
    n_rare_types   = len(rare_types)
    n_rare_cards   = int(df["subtype"].isin(rare_types).sum())

    if n_rare_types:
        logger.info(
            "Bucketing {:,} rare subtype(s) (< {:,} occurrences) into 'Other' "
            "— {:,} card(s) affected",
            n_rare_types, _SUBTYPE_MIN_COUNT, n_rare_cards,
        )
        logger.debug("Subtypes being bucketed: {}", sorted(rare_types.tolist()))
        df["subtype"] = df["subtype"].where(~df["subtype"].isin(rare_types), other="Other")
    else:
        logger.debug(
            "No rare subtypes to bucket — all {:,} subtype(s) have ≥ {:,} occurrences",
            n_types_before, _SUBTYPE_MIN_COUNT,
        )

    n_types_after = df["subtype"].nunique()
    logger.info(
        "Subtype cardinality: {:,} → {:,} unique values",
        n_types_before, n_types_after,
    )
    logger.debug(
        "Subtype distribution after bucketing:\n{}",
        df["subtype"].value_counts().to_string(),
    )
    return df


def load_raw(path: Path) -> pd.DataFrame:
    logger.debug("Reading raw data from {}", path)
    df = pd.read_csv(path)
    logger.info("Loaded {:,} rows, {:,} columns", len(df), len(df.columns))
    return df


def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("Columns not found in source, will be skipped: {}", missing)
        cols = [c for c in cols if c in df.columns]
    df = df[cols]
    logger.debug("Selected {:,} columns: {}", len(cols), list(df.columns))
    return df


def cast_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Cast power and toughness from object (string) dtype to float64.

    After exclude_non_numeric_stats, all remaining values are either null or
    valid numeric strings (e.g. '3', '4'). This step is purely a dtype conversion
    so that downstream arithmetic (stat ratios, comparisons) works correctly.
    """
    logger.debug("Casting power/toughness to numeric dtype")
    for col in ("power", "toughness"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.debug(
        "power dtype: {}, toughness dtype: {}",
        df["power"].dtype, df["toughness"].dtype,
    )
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
        df = exclude_non_numeric_stats(df)
        df = cast_numeric_stats(df)
        df = exclude_x_costs(df)
        df = exclude_hybrid_costs(df)
        df = engineer_type_features(df)
        df = fill_null_subtypes(df)
        df = bucket_rare_subtypes(df)
        df = engineer_mana_features(df)
        df = engineer_stat_ratios(df)

        if len(df) < 100:
            logger.warning(
                "Only {:,} rows remain after preprocessing — "
                "model quality may be poor with so little training data",
                len(df),
            )
        else:
            logger.info("Preprocessing complete: {:,} rows, {:,} columns", *df.shape)

        save(df, PROCESSED_PATH)
        ctx.clean = df
        return ctx
