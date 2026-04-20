from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    _apply_encoding,
    _clean,
    _fit_target_encoding,
    _impute,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "city":             ["Warszawa", "Warszawa", "Kraków", "Kraków", "Łódź"],
        "price":            [500_000, 800_000, 400_000, 600_000, 250_000],
        "price_per_m2":     [10_000, 12_000, 9_000, 11_000, 7_000],
        "area_m2":          [50.0, 65.0, 44.0, 55.0, 36.0],
        "rooms":            [2, 3, 2, 3, 1],
        "floor":            [1, 4, 2, 3, 0],
        "neighborhood":     ["Mokotów", None, "Krowodrza", "Stare Miasto", "Bałuty"],
        "sub_neighborhood": [None, None, None, None, None],
        "is_private_owner": [False, True, False, False, True],
        "url":              ["u1", "u2", "u3", "u4", "u5"],
        "log_price":        np.log1p([500_000, 800_000, 400_000, 600_000, 250_000]),
    })


def test_clean_removes_url_and_sub_neighborhood(raw_df):
    df = raw_df.drop(columns=["log_price"])
    cleaned = _clean(df)
    assert "url" not in cleaned.columns
    assert "sub_neighborhood" not in cleaned.columns


def test_clean_adds_log_price(raw_df):
    df = raw_df.drop(columns=["log_price"])
    cleaned = _clean(df)
    assert "log_price" in cleaned.columns
    assert (cleaned["log_price"] > 0).all()


def test_clean_filters_price_range(raw_df):
    df = raw_df.drop(columns=["log_price"])
    df.loc[0, "price"] = 10_000  # below min
    df.loc[1, "price"] = 9_000_000  # above max
    cleaned = _clean(df)
    assert len(cleaned) == 3


def test_clean_recalculates_price_per_m2(raw_df):
    df = raw_df.drop(columns=["log_price"])
    cleaned = _clean(df)
    expected = cleaned["price"] / cleaned["area_m2"]
    pd.testing.assert_series_equal(
        cleaned["price_per_m2"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_impute_fills_neighborhood_with_city(raw_df):
    imputed = _impute(raw_df.copy())
    # row 1 had null neighborhood → should be filled with city name
    assert imputed.loc[1, "neighborhood"] == "Warszawa"


def test_impute_fills_rooms_floor_with_median(raw_df):
    df = raw_df.copy()
    df.loc[0, "rooms"] = None
    df.loc[0, "floor"] = None
    imputed = _impute(df)
    assert not imputed["rooms"].isna().any()
    assert not imputed["floor"].isna().any()


def test_target_encoding_unknown_key(raw_df):
    enc_city, enc_neighborhood = _fit_target_encoding(raw_df)
    assert "__unknown__" in enc_city
    assert "__unknown__" in enc_neighborhood


def test_target_encoding_values_are_floats(raw_df):
    enc_city, enc_neighborhood = _fit_target_encoding(raw_df)
    for v in enc_city.values():
        assert isinstance(v, float)


def test_apply_encoding_unknown_city(raw_df):
    enc_city, enc_neighborhood = _fit_target_encoding(raw_df)
    df = raw_df.copy()
    df.loc[0, "city"] = "Nieznane Miasto"
    encoded = _apply_encoding(df, enc_city, enc_neighborhood)
    assert encoded.loc[0, "city_enc"] == enc_city["__unknown__"]
