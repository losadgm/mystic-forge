from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests
from mystic_forge.logger import logger

if TYPE_CHECKING:
    from mystic_forge.pipeline import Context

MTGJSON_URL = "https://mtgjson.com/api/v5/FDN.json"
RAW_PATH = Path("data/raw/cards_seed.csv")


def fetch_cards() -> list[dict]:
    logger.info("Fetching card data from {}", MTGJSON_URL)
    response = requests.get(MTGJSON_URL, timeout=30)
    response.raise_for_status()
    cards = response.json()["data"]["cards"]
    logger.info("Fetched {:,} cards successfully", len(cards))
    return cards


def build_dataframe(cards: list[dict]) -> pd.DataFrame:
    logger.debug("Building DataFrame from {:,} cards", len(cards))
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
    logger.debug("DataFrame shape: {} rows x {} columns", *df.shape)
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.success("Saved {:,} rows to {}", len(df), path)


class FetchCardsStage:
    name = "fetch_cards"

    def run(self, ctx: Context) -> Context:
        cards = fetch_cards()
        df = build_dataframe(cards)
        save(df, RAW_PATH)
        ctx.raw = df
        return ctx
