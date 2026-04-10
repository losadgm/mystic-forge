import os

import pandas as pd
import requests
from logger import logger

MTGJSON_URL = "https://mtgjson.com/api/v5/FDN.json"

def fetch_cards() -> list[dict]:
    logger.info("Fetching card data from {}", MTGJSON_URL)
    response = requests.get(MTGJSON_URL, timeout=30)
    response.raise_for_status()
    cards = response.json()["data"]["cards"]
    logger.info("Fetched {} cards successfully", len(cards))
    return cards


def build_dataframe(cards: list[dict]) -> pd.DataFrame:
    logger.debug("Building DataFrame from card data")
    return pd.DataFrame([{
        "name":      c.get("name"),
        "type":      c.get("type"),
        "mana_cost": c.get("manaCost"),
        "power":     c.get("power"),
        "toughness": c.get("toughness"),
        "rarity":    c.get("rarity"),
        "colors":    ",".join(c.get("colors", [])),
        "text":      c.get("text"),
    } for c in cards])


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved {} rows to {}", len(df), path)


if __name__ == "__main__":
    try:
        cards = fetch_cards()
        df = build_dataframe(cards)
        save_csv(df, "data/raw/cards_seed.csv")
    except requests.HTTPError as e:
        logger.error("HTTP error fetching data: {}", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error: {}", e)
        raise
