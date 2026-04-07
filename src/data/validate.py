from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "otodom_all.csv"

REQUIRED_COLUMNS = {
    "city", "price", "price_per_m2", "area_m2",
    "rooms", "floor", "neighborhood", "is_private_owner",
}

MIN_RECORDS_PER_CITY = 300


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(csv_path: Path = DATA_PATH) -> bool:
    """Validate raw CSV. Returns True if all checks pass, False otherwise."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not csv_path.exists():
        log.error("File not found: %s", csv_path)
        return False

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    log.info("Loaded %d rows from %s", len(df), csv_path)

    ok = True

    # --- schema check ---
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        log.error("Missing columns: %s", missing_cols)
        ok = False
    else:
        log.info("Schema OK — all required columns present")

    # --- record counts per city ---
    print("\n--- Record count per city ---")
    counts = df["city"].value_counts()
    for city, n in counts.items():
        flag = "  ⚠ LOW" if n < MIN_RECORDS_PER_CITY else ""
        print(f"  {city:<15} {n:>6}{flag}")
    print(f"  {'TOTAL':<15} {len(df):>6}")

    low_cities = counts[counts < MIN_RECORDS_PER_CITY]
    if not low_cities.empty:
        log.warning("Cities below %d records: %s", MIN_RECORDS_PER_CITY, low_cities.index.tolist())

    # --- null rates ---
    print("\n--- Null % per column ---")
    null_pct = df.isna().mean().mul(100).round(1)
    for col, pct in null_pct.items():
        flag = "  ⚠ HIGH" if pct > 20 else ""
        print(f"  {col:<20} {pct:>5.1f}%{flag}")

    high_null = null_pct[null_pct > 20]
    if not high_null.empty:
        log.warning("Columns with >20%% nulls: %s", high_null.index.tolist())

    if ok:
        log.info("Validation PASSED")
    else:
        log.error("Validation FAILED")

    return ok


if __name__ == "__main__":
    passed = validate()
    sys.exit(0 if passed else 1)
