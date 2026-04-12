from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

MTGJSON_BASE_URL = "https://mtgjson.com/api/v5/{set_code}.json"
RAW_PATH         = Path("data/raw/cards_seed.csv")

# Set codes to fetch and merge into the training corpus.
# Each code maps to one JSON file on mtgjson.com (e.g. "FDN" → FDN.json).
# Add or remove codes here to change which sets are included.
SETS = [
    "FDN",   # Foundations        (2024)
    "MKM",   # Murders at Karlov  (2024)
    "BLB",   # Bloomburrow        (2024)
    "DSK",   # Duskmourn          (2024)
]


def fetch_set(set_code: str) -> list[dict]:
    """Fetch all cards for a single set from mtgjson.

    Returns an empty list (instead of raising) on network or structure errors
    so that one bad set does not abort the whole fetch.
    """
    url = MTGJSON_BASE_URL.format(set_code=set_code)
    logger.info("Fetching set '{}' from {}", set_code, url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch set '{}': {} — skipping", set_code, exc)
        return []

    if "data" not in data or "cards" not in data.get("data", {}):
        logger.warning("Unexpected response structure for set '{}' — missing 'data.cards', skipping", set_code)
        return []

    cards = data["data"]["cards"]
    logger.info("Set '{}': {:,} cards fetched", set_code, len(cards))
    return cards


def fetch_all(set_codes: list[str]) -> list[dict]:
    """Fetch cards from every set in set_codes and return them as a flat list."""
    all_cards: list[dict] = []
    for code in set_codes:
        all_cards.extend(fetch_set(code))
    logger.info("Total cards fetched across {:,} set(s): {:,}", len(set_codes), len(all_cards))
    return all_cards


def build_dataframe(cards: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from raw card dicts and deduplicate by card name.

    The same card can appear in multiple sets with identical stats, so we keep
    only the first occurrence (preserving list order, which reflects SETS order).
    """
    logger.debug("Building DataFrame from {:,} raw card records", len(cards))
    df = pd.DataFrame([{
        "name":      c.get("name"),
        "type":      c.get("type"),
        "mana_cost": c.get("manaCost"),
        "power":     c.get("power"),
        "toughness": c.get("toughness"),
        "rarity":    c.get("rarity"),
        "colors":    ",".join(c.get("colors", [])) or None,
        "text":      c.get("text"),
    } for c in cards])

    before = len(df)
    df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
    duplicates = before - len(df)
    logger.info(
        "Deduplicated by name: {:,} duplicate(s) dropped — {:,} unique cards",
        duplicates, len(df),
    )
    logger.debug("DataFrame shape: {} rows x {} columns", *df.shape)

    # Warn only for columns that should always be populated.
    # power/toughness/text/mana_cost being sparse is expected (non-creature cards).
    always_populated = ["name", "type", "rarity"]
    for col in always_populated:
        if col not in df.columns:
            continue
        pct = df[col].isna().mean()
        if pct > 0:
            logger.warning("Column '{}' is {:.0%} null after fetch — data may be incomplete", col, pct)

    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.success("Saved {:,} rows to {}", len(df), path)


class FetchStage:
    name = "fetch"

    def run(self, ctx: Context) -> Context:
        cards = fetch_all(SETS)
        if not cards:
            logger.error("No cards fetched from any set — aborting pipeline")
            raise RuntimeError("FetchStage: no cards fetched")
        df = build_dataframe(cards)
        save(df, RAW_PATH)
        ctx.raw = df
        return ctx


def main() -> None:
    logger.info("=== fetch start ===")
    cards = fetch_all(SETS)
    if not cards:
        logger.error("No cards fetched from any set")
        raise SystemExit(1)
    df = build_dataframe(cards)
    save(df, RAW_PATH)
    logger.info("=== fetch done ===")


if __name__ == "__main__":
    main()
