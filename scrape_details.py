from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_PATH  = Path(__file__).parent / "data" / "snapshots" / "2026-04_merged_36318rows.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "raw" / "otodom_details.csv"

SAVE_EVERY = 100   # write to disk every N scraped listings
DELAY_LO   = 2.0
DELAY_HI   = 4.5

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.otodom.pl/",
}

# Characteristics keys to extract from listing __NEXT_DATA__
CHAR_KEYS: dict[str, str] = {
    "build_year":            "rok_budowy",
    "building_type":         "rodzaj_zabudowy",
    "heating":               "ogrzewanie",
    "construction_status":   "stan_wykonczenia",
    "building_ownership":    "forma_wlasnosci",
    "building_floors_num":   "liczba_pieter",
    "windows_type":          "okna",
    "market":                "rynek",
    "rent":                  "czynsz",
    "free_from":             "dostepne_od",
}

# Features (boolean flags) — exact strings from listing["features"]
FEATURE_FLAGS: list[str] = [
    "winda",
    "balkon",
    "taras",
    "ogródek",
    "piwnica",
    "garaż/miejsce parkingowe",
    "dwupoziomowe",
    "oddzielna kuchnia",
    "klimatyzacja",
    "meble",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def fetch_listing(session: requests.Session, url: str) -> BeautifulSoup | None:
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            log.warning("HTTP %d for %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.RequestException as exc:
            log.warning("Request error: %s (attempt %d)", exc, attempt + 1)
        time.sleep(random.uniform(5.0, 9.0))
    return None


def parse_details(soup: BeautifulSoup) -> dict[str, Any]:
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return {}

    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return {}

    ad = (
        data.get("props", {})
            .get("pageProps", {})
            .get("ad", {})
    )
    if not ad:
        return {}

    result: dict[str, Any] = {}

    # Characteristics — key/value pairs
    for char in ad.get("characteristics", []):
        key     = char.get("key", "")
        value   = char.get("value")
        col     = CHAR_KEYS.get(key)
        if col:
            result[col] = value

    # Feature flags — list of strings from listing page
    features = set(ad.get("features", []))
    col_map = {
        "winda":                        "winda",
        "balkon":                       "balkon",
        "taras":                        "taras",
        "ogródek":                      "ogrodek",
        "piwnica":                      "piwnica",
        "garaż/miejsce parkingowe":     "garaz",
        "dwupoziomowe":                 "dwupoziomowe",
        "oddzielna kuchnia":            "oddzielna_kuchnia",
        "klimatyzacja":                 "klimatyzacja",
        "meble":                        "umeblowane",
        "teren zamknięty":              "teren_zamkniety",
        "monitoring / ochrona":         "monitoring",
        "domofon / wideofon":           "domofon",
        "drzwi / okna antywłamaniowe":  "okna_antywlamaniowe",
        "internet":                     "internet",
        "pralka":                       "pralka",
        "zmywarka":                     "zmywarka",
        "lodówka":                      "lodowka",
        "kuchenka":                     "kuchenka",
        "piekarnik":                    "piekarnik",
        "telewizor":                    "telewizor",
    }
    for flag, col in col_map.items():
        result[col] = flag in features

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    log.info("Loaded %d listings from %s", len(df), INPUT_PATH.name)

    urls = df["url"].dropna().tolist()

    # Resume: skip already scraped URLs
    already_done: set[str] = set()
    existing_rows: list[dict] = []

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig")
        already_done = set(existing["url"].dropna().tolist())
        existing_rows = existing.to_dict("records")
        log.info("Resuming — %d already scraped, %d remaining",
                 len(already_done), len(urls) - len(already_done))

    to_scrape = [u for u in urls if u not in already_done]
    log.info("Scraping details for %d listings", len(to_scrape))

    session = requests.Session()
    session.headers.update(HEADERS)

    new_rows: list[dict] = []
    total_done = len(already_done)

    for i, url in enumerate(to_scrape, 1):
        soup = fetch_listing(session, url)
        if soup is None:
            log.warning("Failed to fetch %s — skipping", url)
            continue

        details = parse_details(soup)
        details["url"] = url
        new_rows.append(details)
        total_done += 1

        log.info(
            "[%d/%d] %s — extracted %d fields",
            total_done, len(urls), url.split("/")[-1], len(details) - 1,
        )

        if len(new_rows) % SAVE_EVERY == 0:
            _flush(existing_rows + new_rows)
            log.info("Checkpoint saved — %d rows total", len(existing_rows) + len(new_rows))

        time.sleep(random.uniform(DELAY_LO, DELAY_HI))

    _flush(existing_rows + new_rows)
    log.info("Done. %d detail rows saved → %s", len(existing_rows) + len(new_rows), OUTPUT_PATH)


def _flush(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
