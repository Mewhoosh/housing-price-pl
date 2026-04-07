from __future__ import annotations

import numpy as np
import pytest

from src.features.build_features import DATA_PATH, build_features


@pytest.mark.skipif(not DATA_PATH.exists(), reason="otodom_all.csv not available in CI")
def test_build_features_shapes():
    feats = build_features()
    assert feats.X_train.shape[0] > 0
    assert feats.X_test.shape[0] > 0
    assert feats.X_train.shape[1] == 6
    assert len(feats.enc_city) > 1
    assert len(feats.enc_neighborhood) > 1


@pytest.mark.skipif(not DATA_PATH.exists(), reason="otodom_all.csv not available in CI")
def test_build_features_no_nulls():
    feats = build_features()
    assert not feats.X_train.isna().any().any()
    assert not feats.X_test.isna().any().any()


@pytest.mark.skipif(not DATA_PATH.exists(), reason="otodom_all.csv not available in CI")
def test_target_encoding_unknown_fallback():
    feats = build_features()
    assert "__unknown__" in feats.enc_city
    assert "__unknown__" in feats.enc_neighborhood
