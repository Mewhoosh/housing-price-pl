from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# City-level slugs — one entry per city, always scraped first.
# Catches listings that Otodom does not tag to a specific district.
CITY_SLUGS: dict[str, str] = {
    "warszawa":  "Warszawa",
    "krakow":    "Kraków",
    "wroclaw":   "Wrocław",
    "poznan":    "Poznań",
    "gdansk":    "Gdańsk",
    "lodz":      "Łódź",
    "szczecin":  "Szczecin",
    "bydgoszcz": "Bydgoszcz",
    "lublin":    "Lublin",
    "katowice":  "Katowice",
    "bialystok": "Białystok",
    "rzeszow":   "Rzeszów",
    "kielce":    "Kielce",
    "olsztyn":   "Olsztyn",
    "torun":     "Toruń",
}

# District-level slugs — scraped in addition to the city-level slug above.
# Bypasses Otodom's ~25-page pagination limit for large cities.
# The city-level catch-all is already covered by CITY_SLUGS.
DISTRICT_SLUGS: dict[str, list[str]] = {
    "Warszawa": [
        "mazowieckie/warszawa/bemowo",
        "mazowieckie/warszawa/bialoleka",
        "mazowieckie/warszawa/bielany",
        "mazowieckie/warszawa/mokotow",
        "mazowieckie/warszawa/ochota",
        "mazowieckie/warszawa/praga-polnoc",
        "mazowieckie/warszawa/praga-poludnie",
        "mazowieckie/warszawa/rembertow",
        "mazowieckie/warszawa/srodmiescie",
        "mazowieckie/warszawa/targowek",
        "mazowieckie/warszawa/ursus",
        "mazowieckie/warszawa/ursynow",
        "mazowieckie/warszawa/wawer",
        "mazowieckie/warszawa/wesola",
        "mazowieckie/warszawa/wilanow",
        "mazowieckie/warszawa/wlochy",
        "mazowieckie/warszawa/wola",
        "mazowieckie/warszawa/zoliborz",
    ],
    "Kraków": [
        "malopolskie/krakow/bienczyce",
        "malopolskie/krakow/biezanow-prokocim",
        "malopolskie/krakow/bronowice",
        "malopolskie/krakow/czyzyny",
        "malopolskie/krakow/debniki",
        "malopolskie/krakow/grzegorzki",
        "malopolskie/krakow/krowodrza",
        "malopolskie/krakow/lagiewniki-borek-falecki",
        "malopolskie/krakow/mistrzejowice",
        "malopolskie/krakow/nowa-huta",
        "malopolskie/krakow/podgorze",
        "malopolskie/krakow/podgorze-duchackie",
        "malopolskie/krakow/pradnik-bialy",
        "malopolskie/krakow/pradnik-czerwony",
        "malopolskie/krakow/stare-miasto",
        "malopolskie/krakow/swoszowice",
        "malopolskie/krakow/wzgorza-krzeslawickie",
        "malopolskie/krakow/zwierzyniec",
    ],
    "Wrocław": [
        "dolnoslaskie/wroclaw/fabryczna",
        "dolnoslaskie/wroclaw/krzyki",
        "dolnoslaskie/wroclaw/psie-pole",
        "dolnoslaskie/wroclaw/stare-miasto",
        "dolnoslaskie/wroclaw/srodmiescie",
    ],
    "Poznań": [
        "wielkopolskie/poznan/grunwald",
        "wielkopolskie/poznan/jezyce",
        "wielkopolskie/poznan/nowe-miasto",
        "wielkopolskie/poznan/stare-miasto",
        "wielkopolskie/poznan/wilda",
    ],
    "Gdańsk": [
        "pomorskie/gdansk/chelm",
        "pomorskie/gdansk/morena",
        "pomorskie/gdansk/oliwa",
        "pomorskie/gdansk/przymorze-wielkie",
        "pomorskie/gdansk/srodmiescie",
        "pomorskie/gdansk/wrzeszcz",
        "pomorskie/gdansk/zaspa-mlyniec",
    ],
    "Łódź": [
        "lodzkie/lodz/baluty",
        "lodzkie/lodz/gorna",
        "lodzkie/lodz/polesie",
        "lodzkie/lodz/srodmiescie",
        "lodzkie/lodz/widzew",
    ],
}

BASE_URL   = "https://www.otodom.pl/pl/oferty/sprzedaz/mieszkanie/{city}"
MAX_PAGES  = 25
DELAY_LO   = 2.0
DELAY_HI   = 4.5
OUTPUT_DIR = Path(__file__).parent / "data" / "raw"

ROOMS_MAP: dict[str, int] = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX_OR_MORE": 6,
}
FLOOR_MAP: dict[str, int] = {
    "GROUND": 0, "FIRST": 1, "SECOND": 2, "THIRD": 3,
    "FOURTH": 4, "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7,
    "EIGHTH": 8, "NINTH": 9, "TENTH_OR_ABOVE": 10,
}

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Listing:
    city:             str
    price:            int | None
    price_per_m2:     float | None
    area_m2:          float | None
    rooms:            int | None
    floor:            int | None
    neighborhood:     str | None
    sub_neighborhood: str | None
    is_private_owner: bool | None
    url:              str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict[str, Any], *keys: str) -> Any:
    """Traverse nested dict safely; return None if any key is missing."""
    node: Any = d
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _get_rev_geo(locations: list[dict], level: str) -> str | None:
    """Extract named location from reverseGeocoding list by locationLevel."""
    for loc in locations or []:
        if loc.get("locationLevel") == level:
            return loc.get("name")
    return None


def _to_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_float(val: Any) -> float | None:
    try:
        return float(str(val).replace(",", ".").replace(" ", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_page(
    session: requests.Session,
    city_slug: str,
    page: int,
) -> BeautifulSoup | None:
    url = BASE_URL.format(city=city_slug)
    params: dict[str, Any] = {"page": page}

    for attempt in range(2):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            log.warning(
                "HTTP %d for %s page %d (attempt %d)",
                resp.status_code, city_slug, page, attempt + 1,
            )
        except requests.RequestException as exc:
            log.warning("Request error: %s (attempt %d)", exc, attempt + 1)

        time.sleep(random.uniform(5.0, 9.0))

    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_listings(soup: BeautifulSoup, city_name: str) -> list[Listing]:
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return []

    try:
        data: dict[str, Any] = json.loads(tag.string)
    except json.JSONDecodeError:
        log.error("Failed to parse __NEXT_DATA__ JSON")
        return []

    items: list[dict] = (
        _safe_get(data, "props", "pageProps", "data", "searchAds", "items") or []
    )

    results: list[Listing] = []
    for item in items:
        if item.get("estate") != "FLAT":
            continue

        price        = _to_int(_safe_get(item, "totalPrice", "value"))
        price_per_m2 = _to_float(_safe_get(item, "pricePerSquareMeter", "value"))
        area_m2      = _to_float(item.get("areaInSquareMeters"))
        rooms        = ROOMS_MAP.get(item.get("roomsNumber") or "", None)
        floor        = FLOOR_MAP.get(item.get("floorNumber") or "", None)

        rev_geo          = _safe_get(item, "location", "reverseGeocoding", "locations") or []
        neighborhood     = _get_rev_geo(rev_geo, "district")
        sub_neighborhood = _get_rev_geo(rev_geo, "residential")
        is_private_owner = item.get("isPrivateOwner")

        slug        = item.get("slug") or ""
        listing_url = f"https://www.otodom.pl/pl/oferta/{slug}" if slug else None

        results.append(Listing(
            city=city_name,
            price=price,
            price_per_m2=price_per_m2,
            area_m2=area_m2,
            rooms=rooms,
            floor=floor,
            neighborhood=neighborhood,
            sub_neighborhood=sub_neighborhood,
            is_private_owner=is_private_owner,
            url=listing_url,
        ))

    return results


# ---------------------------------------------------------------------------
# City scraper
# ---------------------------------------------------------------------------

def scrape_city(
    session: requests.Session,
    city_slug: str,
    city_name: str,
    max_pages: int = MAX_PAGES,
) -> list[Listing]:
    all_listings: list[Listing] = []

    for page in range(1, max_pages + 1):
        soup = fetch_page(session, city_slug, page)
        if soup is None:
            log.warning("[%s] page %d fetch failed — stopping slug", city_name, page)
            break

        listings = parse_listings(soup, city_name)
        if not listings:
            log.info("[%s] page %d returned 0 listings — end of results", city_name, page)
            break

        all_listings.extend(listings)
        log.info("[%s] slug=%s page=%d — %d listings (+%d total)",
                 city_name, city_slug, page, len(listings), len(all_listings))

        time.sleep(random.uniform(DELAY_LO, DELAY_HI))

    return all_listings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: list[dict] = []

    # Phase 1 — city-level slugs for all 15 cities
    log.info("Phase 1: city-level scraping (%d cities)", len(CITY_SLUGS))
    for slug, name in CITY_SLUGS.items():
        log.info("=== [%s] city slug: %s ===", name, slug)
        listings = scrape_city(session, slug, name)
        if not listings:
            log.warning("[%s] no listings — skipping", name)
            continue
        all_rows.extend(asdict(lst) for lst in listings)
        log.info("[%s] +%d rows (running total: %d)", name, len(listings), len(all_rows))

    # Phase 2 — district-level slugs for configured cities
    log.info("Phase 2: district-level scraping (%d cities)", len(DISTRICT_SLUGS))
    for city_name, slugs in DISTRICT_SLUGS.items():
        log.info("=== [%s] %d district slugs ===", city_name, len(slugs))
        for slug in slugs:
            listings = scrape_city(session, slug, city_name)
            if not listings:
                log.warning("[%s] slug=%s — no listings", city_name, slug)
                continue
            all_rows.extend(asdict(lst) for lst in listings)

    if not all_rows:
        log.error("No data scraped.")
        return

    combined = pd.DataFrame(all_rows)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    log.info("Deduplicated: %d → %d rows (removed %d duplicates)",
             before, len(combined), before - len(combined))

    # Save per-city CSVs
    name_to_slug = {v: k for k, v in CITY_SLUGS.items()}
    for city_name, city_df in combined.groupby("city"):
        file_slug = name_to_slug.get(city_name, city_name.lower())
        out_path = OUTPUT_DIR / f"otodom_{file_slug}.csv"
        city_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info("[%s] saved %d rows → %s", city_name, len(city_df), out_path)

    combined_path = OUTPUT_DIR / "otodom_all.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    log.info("Saved combined dataset (%d rows) → %s", len(combined), combined_path)

    # Summary
    counts = combined["city"].value_counts()
    print("\n--- Record count per city ---")
    for city, n in counts.items():
        print(f"  {city:<15} {n:>6}")
    print(f"  {'TOTAL':<15} {len(combined):>6}")

    null_pct = combined.isna().mean().mul(100).round(1)
    print("\n--- Null % per column ---")
    print(null_pct.to_string())


if __name__ == "__main__":
    main()
