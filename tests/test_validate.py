from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import REQUIRED_COLUMNS, validate


def test_validate_passes_on_valid_csv(tmp_path):
    csv = tmp_path / "otodom_all.csv"
    df = pd.DataFrame({col: ["x"] for col in REQUIRED_COLUMNS})
    df.to_csv(csv, index=False)
    assert validate(csv) is True


def test_validate_fails_on_missing_column(tmp_path):
    csv = tmp_path / "otodom_all.csv"
    cols = list(REQUIRED_COLUMNS - {"price"})
    df = pd.DataFrame({col: ["x"] for col in cols})
    df.to_csv(csv, index=False)
    assert validate(csv) is False


def test_validate_fails_on_missing_file(tmp_path):
    assert validate(tmp_path / "nonexistent.csv") is False
