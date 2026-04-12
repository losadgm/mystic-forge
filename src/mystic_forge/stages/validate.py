from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

# Known domain values — any value outside these sets is a rule violation.
_VALID_RARITIES  = frozenset({"common", "uncommon", "rare", "mythic"})
_VALID_CARD_TYPES = frozenset({
    "Creature", "Planeswalker", "Battle", "Enchantment",
    "Artifact", "Land", "Instant", "Sorcery", "Other",
})

# Structural limits for sanity checks.
_MAX_CMC     = 16
_MAX_STAT    = 20

# How far a distribution metric can drift from the training baseline
# before we emit a warning.
_DIST_WARN_THRESHOLD = 0.15   # 15 percentage-point shift in a category


# ---------------------------------------------------------------------------
# Domain rule checks — log warnings for every violation found
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame, required: list[str]) -> int:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: {}", missing)
    else:
        logger.debug("All required columns present")
    return len(missing)


def check_nulls(df: pd.DataFrame, required_non_null: list[str]) -> int:
    violations = 0
    for col in required_non_null:
        if col not in df.columns:
            continue
        null_frac = df[col].isna().mean()
        if null_frac > 0:
            logger.warning(
                "Column '{}' has {:.1%} null values ({:,} / {:,} rows)",
                col, null_frac, int(df[col].isna().sum()), len(df),
            )
            violations += 1
        else:
            logger.debug("Column '{}' — no nulls", col)
    return violations


def check_categorical_values(df: pd.DataFrame, col: str, valid: frozenset[str]) -> int:
    if col not in df.columns:
        return 0
    invalid_mask = ~df[col].isin(valid)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        invalid_vals = df.loc[invalid_mask, col].value_counts().to_dict()
        logger.warning(
            "Column '{}': {:,} rows have unexpected values — {}",
            col, invalid_count, invalid_vals,
        )
    else:
        logger.debug("Column '{}' — all values within expected set", col)
    return invalid_count


def check_numeric_range(df: pd.DataFrame, col: str, min_val: int, max_val: int) -> int:
    if col not in df.columns:
        return 0
    out_of_range = df[col].dropna()
    out_of_range = out_of_range[(out_of_range < min_val) | (out_of_range > max_val)]
    count = len(out_of_range)
    if count:
        logger.warning(
            "Column '{}': {:,} rows outside expected range [{}, {}] "
            "— min: {}, max: {}",
            col, count, min_val, max_val,
            int(df[col].min()), int(df[col].max()),
        )
    else:
        logger.debug(
            "Column '{}' — all values in [{}, {}]", col, min_val, max_val,
        )
    return count


def check_subtype_card_type(synthetic: pd.DataFrame, training: pd.DataFrame) -> int:
    """Count cards where the subtype never co-occurs with that card_type in training.

    Uses the training data as the ground truth for valid combinations rather than
    hard-coding MTG subtype rules, so it adapts automatically to whatever sets are
    in the corpus.
    """
    if not {"card_type", "subtype"}.issubset(synthetic.columns) or \
       not {"card_type", "subtype"}.issubset(training.columns):
        return 0

    valid_subtypes: dict[str, set] = (
        training.groupby("card_type")["subtype"].apply(set).to_dict()
    )

    invalid_mask = ~synthetic.apply(
        lambda row: row["subtype"] in valid_subtypes.get(row["card_type"], set()),
        axis=1,
    )
    count = int(invalid_mask.sum())

    if count:
        top_combos = (
            synthetic.loc[invalid_mask, ["card_type", "subtype"]]
            .value_counts()
            .head(10)
        )
        logger.warning(
            "{:,} cards ({:.1%}) have a subtype not seen with their card_type in training",
            count, count / len(synthetic),
        )
        logger.debug("Top invalid combinations:\n{}", top_combos.to_string())
    else:
        logger.debug("All subtype/card_type combinations present in training")
    return count


def check_stats_symmetry(df: pd.DataFrame) -> int:
    """Verify that power and toughness are either both present or both absent."""
    if "power" not in df.columns or "toughness" not in df.columns:
        return 0
    asymmetric = df["power"].isna() ^ df["toughness"].isna()
    count = int(asymmetric.sum())
    if count:
        logger.warning(
            "{:,} cards have power XOR toughness — stats should be symmetric",
            count,
        )
    else:
        logger.debug("Stats symmetry OK — power and toughness always co-present")
    return count


# ---------------------------------------------------------------------------
# Distribution metrics — compare synthetic output against training baseline
# ---------------------------------------------------------------------------

def compare_distribution(
    synthetic: pd.DataFrame,
    training: pd.DataFrame,
    col: str,
    label: str,
) -> None:
    """Log the distribution of col in synthetic vs training and warn on large drift."""
    if col not in synthetic.columns or col not in training.columns:
        return

    syn_dist   = synthetic[col].value_counts(normalize=True).sort_index()
    train_dist = training[col].value_counts(normalize=True).sort_index()

    # Align on the union of categories.
    all_cats   = syn_dist.index.union(train_dist.index)
    syn_dist   = syn_dist.reindex(all_cats, fill_value=0.0)
    train_dist = train_dist.reindex(all_cats, fill_value=0.0)

    max_drift = float((syn_dist - train_dist).abs().max())
    logger.info(
        "{} distribution — max drift vs training: {:.1%}{}",
        label, max_drift,
        " ⚠" if max_drift > _DIST_WARN_THRESHOLD else "",
    )
    if max_drift > _DIST_WARN_THRESHOLD:
        worst_cat = (syn_dist - train_dist).abs().idxmax()
        logger.warning(
            "  → '{}' drifted most: {:.1%} synthetic vs {:.1%} training",
            worst_cat,
            float(syn_dist[worst_cat]),
            float(train_dist[worst_cat]),
        )
    logger.debug(
        "{} — synthetic:\n{}\ntraining:\n{}",
        label,
        syn_dist.to_string(),
        train_dist.to_string(),
    )


def compare_numeric(
    synthetic: pd.DataFrame,
    training: pd.DataFrame,
    col: str,
    label: str,
) -> None:
    """Log mean/std of col in synthetic vs training."""
    if col not in synthetic.columns or col not in training.columns:
        return
    syn_vals   = synthetic[col].dropna()
    train_vals = training[col].dropna()
    logger.info(
        "{} — synthetic mean: {:.2f} (±{:.2f}), training mean: {:.2f} (±{:.2f})",
        label,
        syn_vals.mean(),   syn_vals.std(),
        train_vals.mean(), train_vals.std(),
    )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class ValidateStage:
    """Domain validation and distribution QA for the synthetic output.

    Does not transform data. Reads ctx.synthetic (produced by SampleStage)
    and ctx.clean (the training baseline) and reports:
      - Rule violations: missing columns, nulls, out-of-range values, etc.
      - Distribution drift: per-column shift between synthetic and training.

    All findings are logged; warnings indicate degraded output quality.
    """

    name = "validate"

    # Columns that must always be present and non-null in the output.
    _REQUIRED = ["card_type", "subtype", "mana_cost", "cmc", "color", "rarity"]

    def run(self, ctx: Context) -> Context:
        if ctx.synthetic is None:
            logger.error("No synthetic data in context — was the sample stage skipped?")
            raise RuntimeError("ValidateStage requires ctx.synthetic")

        df  = ctx.synthetic
        ref = ctx.clean   # may be None if train was skipped

        logger.info("Validating {:,} synthetic rows", len(df))
        total_violations = 0

        # --- Domain rule checks ---
        total_violations += check_required_columns(df, self._REQUIRED)
        total_violations += check_nulls(df, self._REQUIRED)
        total_violations += check_categorical_values(df, "rarity",    _VALID_RARITIES)
        total_violations += check_categorical_values(df, "card_type", _VALID_CARD_TYPES)
        total_violations += check_numeric_range(df, "cmc",       0, _MAX_CMC)
        total_violations += check_numeric_range(df, "power",     0, _MAX_STAT)
        total_violations += check_numeric_range(df, "toughness", 0, _MAX_STAT)
        total_violations += check_stats_symmetry(df)
        if ref is not None:
            total_violations += check_subtype_card_type(df, ref)

        if total_violations == 0:
            logger.info("Domain rules — no violations found")
        else:
            logger.warning("Domain rules — {:,} violation(s) found", total_violations)

        # --- Distribution metrics vs training baseline ---
        if ref is not None:
            logger.info("Comparing distributions against training baseline")
            compare_distribution(df, ref, "color",     "Color")
            compare_distribution(df, ref, "rarity",    "Rarity")
            compare_distribution(df, ref, "card_type", "Card type")
            compare_distribution(df, ref, "subtype",   "Subtype")
            compare_numeric(df, ref, "cmc",   "CMC")
            compare_numeric(df, ref, "power", "Power")
            compare_numeric(df, ref, "toughness", "Toughness")

            # Stats ratio — fraction of cards with power/toughness
            if "power" in df.columns and "power" in ref.columns:
                syn_ratio   = df["power"].notna().mean()
                train_ratio = ref["power"].notna().mean()
                drift       = abs(syn_ratio - train_ratio)
                logger.info(
                    "Stats ratio — synthetic: {:.1%}, training: {:.1%}{}",
                    syn_ratio, train_ratio,
                    " ⚠" if drift > _DIST_WARN_THRESHOLD else "",
                )
        else:
            logger.warning(
                "No training baseline in context — distribution comparison skipped"
            )

        return ctx
