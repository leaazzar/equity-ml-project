"""Tests for SIZE features: market cap, log market cap, relative market size.

`mktcap_millions` here is already unit-normalized ($ millions — see
units.py); tests never feed the raw CRSP `MthCap` column (which is $
thousands) directly, matching how `compute_size_features` is actually
called in the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.size_features import compute_size_features


def test_mktcap_and_log_mktcap() -> None:
    df = pd.DataFrame(
        {
            "mktcap_millions": [100.0, 400.0, np.nan, -5.0],
            "MthCalDt": pd.to_datetime(["2020-01-31"] * 4),
        }
    )
    is_investable = pd.Series([True, True, False, False])
    features = compute_size_features(df, is_investable)

    assert features["size_mktcap"].iloc[0] == pytest.approx(100.0)
    assert features["size_log_mktcap"].iloc[1] == pytest.approx(np.log(400.0))
    assert pd.isna(features["size_mktcap"].iloc[2])
    # Negative/zero market cap is masked, not silently negative-logged.
    assert pd.isna(features["size_log_mktcap"].iloc[3])


def test_relative_mktcap_uses_only_investable_universe_median_that_month() -> None:
    df = pd.DataFrame(
        {
            "mktcap_millions": [100.0, 200.0, 300.0, 100_000.0],
            "MthCalDt": pd.to_datetime(["2020-01-31"] * 4),
        }
    )
    # The 4th row (huge mega-cap) is NOT investable, so it must not distort
    # the reference median used for relative_mktcap.
    is_investable = pd.Series([True, True, True, False])
    features = compute_size_features(df, is_investable)

    # Median of the investable universe (100, 200, 300) is 200.
    assert features["size_relative_mktcap"].iloc[0] == pytest.approx(100.0 / 200.0)
    assert features["size_relative_mktcap"].iloc[2] == pytest.approx(300.0 / 200.0)
    # Even though row 4 is not investable, it still gets a computed value
    # against the universe reference (the raw panel is never filtered).
    assert features["size_relative_mktcap"].iloc[3] == pytest.approx(100_000.0 / 200.0)


def test_relative_mktcap_is_isolated_by_month() -> None:
    df = pd.DataFrame(
        {
            "mktcap_millions": [100.0, 200.0, 1000.0, 2000.0],
            "MthCalDt": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]),
        }
    )
    is_investable = pd.Series([True, True, True, True])
    features = compute_size_features(df, is_investable)

    # January median = 150; February median = 1500 — the two months must not mix.
    assert features["size_relative_mktcap"].iloc[0] == pytest.approx(100.0 / 150.0)
    assert features["size_relative_mktcap"].iloc[2] == pytest.approx(1000.0 / 1500.0)
