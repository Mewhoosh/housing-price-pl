from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT         = Path(__file__).parents[2]
LISTINGS_PATH = ROOT / "data" / "raw" / "otodom_all.csv"
DETAILS_PATH  = ROOT / "data" / "raw" / "otodom_details.csv"

LISTINGS_REQUIRED = {
    "city", "price", "price_per_m2", "area_m2",
    "rooms", "floor", "neighborhood", "is_private_owner", "url",
}
DETAILS_REQUIRED = {
    "url", "rok_budowy", "stan_wykonczenia",
}

MIN_RECORDS_PER_CITY = 300
NULL_WARN_THRESHOLD  = 50  # % — warning only, not failure


def validate() -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not LISTINGS_PATH.exists():
        log.info("Data files not present — skipping validation (CI without data)")
        return True

    ok = True
    ok &= _validate_listings()
    ok &= _validate_details()

    log.info("Validation %s", "PASSED" if ok else "FAILED")
    return ok


def _validate_listings() -> bool:
    df = pd.read_csv(LISTINGS_PATH, encoding="utf-8-sig")
    log.info("Listings: %d rows", len(df))

    ok = _check_schema(df, LISTINGS_REQUIRED, "listings")

    counts = df["city"].value_counts()
    log.info("Record count per city:")
    for city, n in counts.items():
        suffix = " [LOW]" if n < MIN_RECORDS_PER_CITY else ""
        log.info("  %-15s %6d%s", city, n, suffix)
    log.info("  %-15s %6d", "TOTAL", len(df))

    low = counts[counts < MIN_RECORDS_PER_CITY]
    if not low.empty:
        log.warning("Cities below %d records: %s", MIN_RECORDS_PER_CITY, low.index.tolist())

    _log_null_rates(df, "listings")
    return ok


def _validate_details() -> bool:
    if not DETAILS_PATH.exists():
        log.warning("otodom_details.csv not found — skipping details validation")
        return True

    df = pd.read_csv(DETAILS_PATH, encoding="utf-8-sig", low_memory=False)
    log.info("Details: %d rows", len(df))

    ok = _check_schema(df, DETAILS_REQUIRED, "details")
    _log_null_rates(df, "details")
    return ok


def _check_schema(df: pd.DataFrame, required: set[str], name: str) -> bool:
    missing = required - set(df.columns)
    if missing:
        log.error("%s — missing columns: %s", name, sorted(missing))
        return False
    log.info("%s schema OK", name)
    return True


def _log_null_rates(df: pd.DataFrame, name: str) -> None:
    null_pct = df.isna().mean().mul(100).round(1)
    high = null_pct[null_pct > NULL_WARN_THRESHOLD]
    if not high.empty:
        log.warning("%s — columns with >%d%% nulls: %s", name, NULL_WARN_THRESHOLD, high.index.tolist())


if __name__ == "__main__":
    passed = validate()
    sys.exit(0 if passed else 1)
