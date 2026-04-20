from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

DATA_PATH    = Path(__file__).parents[2] / "data" / "raw" / "otodom_all.csv"
DETAILS_PATH = Path(__file__).parents[2] / "data" / "raw" / "otodom_details.csv"

TARGET = "log_price"
SEED   = 17

PRICE_MIN    = 50_000
PRICE_MAX    = 5_000_000
AREA_MIN     = 15.0
AREA_MAX     = 250.0
PRICE_M2_MAX = 40_000.0

BASE_FEATURES = ["area_m2", "rooms", "floor", "city_enc", "neighborhood_enc", "is_private_owner"]

DETAIL_NUMERIC     = ["rok_budowy", "liczba_pieter"]
DETAIL_CATEGORICAL = ["rodzaj_zabudowy", "ogrzewanie", "stan_wykonczenia", "forma_wlasnosci", "rynek", "okna"]
DETAIL_BOOL        = [
    "winda", "balkon", "taras", "ogrodek", "piwnica", "garaz",
    "dwupoziomowe", "oddzielna_kuchnia", "klimatyzacja",
    "teren_zamkniety", "monitoring", "domofon", "okna_antywlamaniowe",
]

# Columns in details CSV not useful for price prediction
DETAIL_DROP = [
    "dostepne_od", "czynsz",
    "pralka", "zmywarka", "lodowka", "kuchenka", "piekarnik",
    "telewizor", "internet", "umeblowane",
]

# Drop rows where these are null — high SHAP importance, imputing adds noise
DETAIL_REQUIRED = ["rok_budowy", "stan_wykonczenia"]

# Impute with mode — low SHAP signal, not worth losing rows
DETAIL_IMPUTE_MODE = ["ogrzewanie", "rodzaj_zabudowy", "forma_wlasnosci", "okna"]

# Impute with median
DETAIL_IMPUTE_MEDIAN = ["liczba_pieter"]


class Features(NamedTuple):
    X_train:            pd.DataFrame
    X_test:             pd.DataFrame
    y_train:            pd.Series
    y_test:             pd.Series
    enc_city:           dict[str, float]
    enc_neighborhood:   dict[str, float]
    city_neighborhoods: dict[str, list[str]]
    feature_columns:    list[str]
    ohe_categories:     dict[str, list[str]]


def build_features(
    csv_path: Path = DATA_PATH,
    details_path: Path = DETAILS_PATH,
) -> Features:
    df = _load_and_merge(csv_path, details_path)
    df = _clean(df)
    df = _impute_base(df)
    df = _apply_null_strategy(df)

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

    train_df, test_df, ohe_categories = _encode_details(train_df, test_df)

    feature_columns = _get_feature_columns(ohe_categories)
    X_train = train_df[feature_columns]
    X_test  = test_df[feature_columns]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    return Features(
        X_train, X_test, y_train, y_test,
        enc_city, enc_neighborhood, city_neighborhoods,
        feature_columns, ohe_categories,
    )


def _load_and_merge(csv_path: Path, details_path: Path) -> pd.DataFrame:
    main    = pd.read_csv(csv_path,     encoding="utf-8-sig")
    details = pd.read_csv(details_path, encoding="utf-8-sig", low_memory=False)
    log.info("Loaded %d listings, %d details", len(main), len(details))

    details = details.drop(columns=DETAIL_DROP, errors="ignore")
    df = main.merge(details, on="url", how="inner")
    log.info("After merge: %d rows", len(df))
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["url", "sub_neighborhood"], errors="ignore")

    before = len(df)
    df = df.dropna(subset=["price", "area_m2"])
    df = df[
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["area_m2"].between(AREA_MIN, AREA_MAX)
    ]
    df["price_per_m2"] = df["price"] / df["area_m2"]
    df = df[df["price_per_m2"] <= PRICE_M2_MAX]
    log.info("Cleaned: %d → %d rows (removed %d)", before, len(df), before - len(df))

    df["log_price"] = np.log1p(df["price"])
    return df.reset_index(drop=True)


def _impute_base(df: pd.DataFrame) -> pd.DataFrame:
    df["neighborhood"] = df["neighborhood"].fillna(df["city"])
    for col in ("rooms", "floor"):
        medians = df.groupby("city")[col].transform("median")
        df[col] = df[col].fillna(medians)
    df["is_private_owner"] = df["is_private_owner"].fillna(False).astype(int)
    return df


def _apply_null_strategy(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=DETAIL_REQUIRED).copy()
    log.info(
        "Dropped %d rows where %s null",
        before - len(df), DETAIL_REQUIRED,
    )

    for col in DETAIL_IMPUTE_MODE:
        if col in df.columns:
            mode = df[col].mode()
            if len(mode):
                df[col] = df[col].fillna(mode[0])

    for col in DETAIL_IMPUTE_MEDIAN:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def _fit_target_encoding(
    train_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    global_mean = float(train_df["log_price"].mean())

    enc_city = train_df.groupby("city")["log_price"].mean().to_dict()
    enc_city["__unknown__"] = global_mean

    enc_neighborhood = train_df.groupby("neighborhood")["log_price"].mean().to_dict()
    enc_neighborhood["__unknown__"] = global_mean

    log.info(
        "Target encoding: %d cities, %d neighborhoods",
        len(enc_city) - 1, len(enc_neighborhood) - 1,
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


def _encode_details(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    ohe_categories: dict[str, list[str]] = {}

    for col in DETAIL_CATEGORICAL:
        if col not in train_df.columns:
            continue
        categories = sorted(train_df[col].dropna().unique().tolist())
        ohe_categories[col] = categories
        for cat in categories:
            col_name = f"{col}_{cat}"
            train_df[col_name] = (train_df[col] == cat).astype(int)
            test_df[col_name]  = (test_df[col]  == cat).astype(int)

    for col in DETAIL_BOOL:
        if col in train_df.columns:
            mapping = {True: 1, False: 0, "True": 1, "False": 0}
            train_df[col] = train_df[col].map(mapping).fillna(0).astype(int)
            test_df[col]  = test_df[col].map(mapping).fillna(0).astype(int)

    return train_df, test_df, ohe_categories


def _get_feature_columns(ohe_categories: dict[str, list[str]]) -> list[str]:
    ohe_cols = [f"{col}_{cat}" for col, cats in ohe_categories.items() for cat in cats]
    return BASE_FEATURES + DETAIL_NUMERIC + ohe_cols + DETAIL_BOOL
