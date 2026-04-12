from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

MODEL_STATS    = Path("output/synth_with_stats.pkl")
MODEL_NO_STATS = Path("output/synth_no_stats.pkl")
SYNTHETIC_PATH = Path("data/synthetic/cards_synthetic.csv")
DEFAULT_ROWS   = 300

# Internal pip/generic columns dropped after mana_cost is reconstructed.
# cmc is kept — it is a useful output column for downstream analysis.
_MANA_COMPONENTS = ["pip_W", "pip_U", "pip_B", "pip_R", "pip_G", "generic"]

# Pip column order must match WUBRG so zip(_PIP_COLS, _COLOR_ORDER) pairs correctly.
_COLOR_ORDER = "WUBRG"
_PIP_COLS    = ["pip_W", "pip_U", "pip_B", "pip_R", "pip_G"]

# Fraction of rows allowed to use fallback values before we emit a warning.
_FALLBACK_WARN_THRESHOLD = 0.2


def load_model(path: Path) -> CopulaGANSynthesizer:
    if not path.exists():
        logger.error("Model file not found: {}", path)
        raise FileNotFoundError(path)
    logger.debug("Loading model from {}", path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CopulaGANSynthesizer.load(str(path))
    return model


def sample(model: CopulaGANSynthesizer, num_rows: int, label: str) -> pd.DataFrame:
    logger.info("Sampling {:,} '{}' rows", num_rows, label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = model.sample(num_rows=num_rows)
    logger.debug("'{}' shape: {} rows x {} columns", label, *df.shape)
    return df


def enforce_mana_invariants(df: pd.DataFrame) -> pd.DataFrame:
    """Clip pip/generic columns and derive cmc and color deterministically.

    cmc is excluded from training so it is absent from the sampled DataFrame
    and must be computed here. color is re-derived from the normalized pips to
    guarantee it is consistent with the mana cost that will be reconstructed.

    Steps:
      1. Clip pip_* and generic to non-negative integers.
      2. Compute cmc  = Σpip_* + generic.
      3. Derive color = WUBRG-ordered symbols where pip_X > 0, or "colorless".
    """
    logger.info("Enforcing mana invariants (clip pips, derive cmc and color)")

    for col in _PIP_COLS + ["generic"]:
        df[col] = df[col].clip(lower=0).round().astype(int)

    df["cmc"] = df[_PIP_COLS].sum(axis=1) + df["generic"]
    logger.debug(
        "CMC — range: {}–{}, mean: {:.1f}",
        int(df["cmc"].min()), int(df["cmc"].max()), df["cmc"].mean(),
    )

    def _derive_color(row: pd.Series) -> str:
        return "".join(sym for col, sym in zip(_PIP_COLS, _COLOR_ORDER) if row[col] > 0) \
               or "colorless"

    df["color"] = df[_PIP_COLS].apply(_derive_color, axis=1)
    logger.debug("Color distribution:\n{}", df["color"].value_counts().to_string())

    return df


def reconstruct_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct integer power/toughness from per-cmc ratios, then drop the ratios.

    Relies on cmc already being set by enforce_mana_invariants.

        power     = round(power_per_cmc     × max(cmc, 1))
        toughness = round(toughness_per_cmc × max(cmc, 1))

    cmc is clipped to a minimum of 1 to handle 0-cost cards (e.g. Ornithopter).
    A NaN ratio is filled with 1.0 as a safe fallback.
    """
    logger.info("Reconstructing power/toughness from per-cmc ratios")
    nan_frac = df["power_per_cmc"].isna().mean()
    if nan_frac > _FALLBACK_WARN_THRESHOLD:
        logger.warning(
            "  → {:.0%} of rows had NaN power_per_cmc — "
            "fallback ratio 1.0 applied; model may have produced unrealistic stat distributions",
            nan_frac,
        )

    effective_cmc = df["cmc"].clip(lower=1)
    df["power"]     = (df["power_per_cmc"].fillna(1.0)     * effective_cmc).round().clip(lower=0).astype(int)
    df["toughness"] = (df["toughness_per_cmc"].fillna(1.0) * effective_cmc).round().clip(lower=0).astype(int)
    logger.debug(
        "Stats reconstructed — power range: {}–{}, toughness range: {}–{}",
        int(df["power"].min()), int(df["power"].max()),
        int(df["toughness"].min()), int(df["toughness"].max()),
    )
    return df.drop(columns=["power_per_cmc", "toughness_per_cmc"])


def reconstruct_mana_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a mana cost string from pip/generic columns, then drop them.

    Relies on pip_* and generic already being non-negative integers (guaranteed
    by enforce_mana_invariants). A card with all components at zero gets "{0}".

    Drops: pip_W, pip_U, pip_B, pip_R, pip_G, generic. Keeps cmc.
    """
    def _build_cost(row: pd.Series) -> str:
        parts = [f"{{{row['generic']}}}"] if row["generic"] else []
        for col, symbol in zip(_PIP_COLS, _COLOR_ORDER):
            parts.extend([f"{{{symbol}}}"] * row[col])
        return "".join(parts) or "{0}"

    logger.info("Reconstructing mana cost strings from pip/generic components")
    df["mana_cost"] = df[["generic"] + _PIP_COLS].apply(_build_cost, axis=1)

    zero_cost_frac = (df["mana_cost"] == "{0}").mean()
    if zero_cost_frac > _FALLBACK_WARN_THRESHOLD:
        logger.warning(
            "  → {:.0%} of cards got cost '{{0}}' — "
            "model may have collapsed pip/generic distributions to near-zero",
            zero_cost_frac,
        )
    logger.debug("Top mana costs in sample: {}", df["mana_cost"].value_counts().head(5).to_dict())
    return df.drop(columns=_MANA_COMPONENTS)


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
        path_stats    = Path(ctx.meta.get("model_stats",    str(MODEL_STATS)))
        path_no_stats = Path(ctx.meta.get("model_no_stats", str(MODEL_NO_STATS)))
        ratio         = ctx.meta.get("stats_ratio")
        if ratio is None:
            logger.warning(
                "stats_ratio not found in pipeline context — "
                "train stage may have been skipped; defaulting to 0.5"
            )
            ratio = 0.5

        n_stats    = math.ceil(DEFAULT_ROWS * ratio)
        n_no_stats = DEFAULT_ROWS - n_stats
        logger.info(
            "Target: {:,} total rows — {:,} with stats ({:.1%}), {:,} without ({:.1%})",
            DEFAULT_ROWS, n_stats, ratio, n_no_stats, 1 - ratio,
        )

        model_stats    = load_model(path_stats)
        model_no_stats = load_model(path_no_stats)

        df_stats    = sample(model_stats,    n_stats,    "with_stats")
        df_no_stats = sample(model_no_stats, n_no_stats, "no_stats")

        df_stats    = enforce_mana_invariants(df_stats)
        df_no_stats = enforce_mana_invariants(df_no_stats)

        df_stats    = reconstruct_stats(df_stats)
        df_stats    = reconstruct_mana_cost(df_stats)
        df_no_stats = reconstruct_mana_cost(df_no_stats)

        df = merge(df_stats, df_no_stats)
        save(df, SYNTHETIC_PATH)

        ctx.synthetic = df
        return ctx
