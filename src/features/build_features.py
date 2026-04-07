from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "otodom_all.csv"

FEATURES = ["area_m2", "rooms", "floor", "city_enc", "neighborhood_enc", "is_private_owner"]
TARGET   = "log_price"
SEED     = 17

PRICE_MIN    = 50_000
PRICE_MAX    = 5_000_000
AREA_MIN     = 15.0
AREA_MAX     = 250.0
PRICE_M2_MAX = 40_000.0


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

class Features(NamedTuple):
    X_train:          pd.DataFrame
    X_test:           pd.DataFrame
    y_train:          pd.Series
    y_test:           pd.Series
    enc_city:         dict[str, float]        # {city: mean_log1p_price, "__unknown__": global_mean}
    enc_neighborhood: dict[str, float]        # {neighborhood: mean_log1p_price, "__unknown__": global_mean}
    city_neighborhoods: dict[str, list[str]]  # {city: sorted list of neighborhoods} — for app.py dropdowns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(csv_path: Path = DATA_PATH) -> Features:
    """Full feature engineering pipeline.

    1. Load raw CSV
    2. Clean and filter
    3. Impute nulls
    4. Train/test split (80/20, SEED=17)
    5. Compute target encoding on train set only (anti-leakage)
    6. Apply encoding to both sets
    7. Return Features namedtuple
    """
    df = _load(csv_path)
    df = _clean(df)
    df = _impute(df)

    # city_neighborhoods from full dataset — needed by app.py dropdowns
    city_neighborhoods: dict[str, list[str]] = (
        df.groupby("city")["neighborhood"]
        .apply(lambda x: sorted(x.dropna().unique().tolist()))
        .to_dict()
    )

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    log.info("Split: %d train / %d test", len(train_df), len(test_df))

    enc_city, enc_neighborhood = _fit_target_encoding(train_df)

    train_df = _apply_encoding(train_df, enc_city, enc_neighborhood)
    test_df  = _apply_encoding(test_df,  enc_city, enc_neighborhood)

    X_train = train_df[FEATURES]
    X_test  = test_df[FEATURES]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    return Features(X_train, X_test, y_train, y_test, enc_city, enc_neighborhood, city_neighborhoods)


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    log.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["url", "sub_neighborhood"], errors="ignore")

    before = len(df)
    df = df.dropna(subset=["price", "area_m2"])
    df = df[
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["area_m2"].between(AREA_MIN, AREA_MAX)
    ]

    # recalculate price_per_m2 from cleaned price/area (more reliable than scraped value)
    df["price_per_m2"] = df["price"] / df["area_m2"]
    df = df[df["price_per_m2"] <= PRICE_M2_MAX]

    log.info("Cleaned: %d → %d rows (removed %d)", before, len(df), before - len(df))

    df["log_price"] = np.log1p(df["price"])
    return df.reset_index(drop=True)


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    df["neighborhood"] = df["neighborhood"].fillna(df["city"])

    for col in ("rooms", "floor"):
        medians = df.groupby("city")[col].transform("median")
        df[col] = df[col].fillna(medians)

    df["is_private_owner"] = df["is_private_owner"].fillna(False).astype(int)
    return df


def _fit_target_encoding(
    train_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean log1p(price) per city and neighborhood on train set only."""
    global_mean = float(train_df["log_price"].mean())

    enc_city = train_df.groupby("city")["log_price"].mean().to_dict()
    enc_city["__unknown__"] = global_mean

    enc_neighborhood = train_df.groupby("neighborhood")["log_price"].mean().to_dict()
    enc_neighborhood["__unknown__"] = global_mean

    log.info(
        "Target encoding fitted: %d cities, %d neighborhoods",
        len(enc_city) - 1,
        len(enc_neighborhood) - 1,
    )
    return enc_city, enc_neighborhood


def _apply_encoding(
    df: pd.DataFrame,
    enc_city: dict[str, float],
    enc_neighborhood: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    df["city_enc"]         = df["city"].map(enc_city).fillna(enc_city["__unknown__"])
    df["neighborhood_enc"] = df["neighborhood"].map(enc_neighborhood).fillna(enc_neighborhood["__unknown__"])
    return df
