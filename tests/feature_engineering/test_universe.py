"""Tests for the investable-universe flag."""

from __future__ import annotations

import pandas as pd

from feature_engineering.config import FeatureConfig
from feature_engineering.universe import build_investable_universe


def _panel(**overrides) -> pd.DataFrame:
    base = {
        "MthPrc": [10.0, 0.5, None, 10.0, 10.0],
        "MthCap": [1000.0, 1000.0, 1000.0, 0.0, 1000.0],
        "MthRetFlg": ["CR", "CR", "CR", "CR", "NT"],
        "SICCD": [2000, 2000, 2000, 2000, 6020],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_excludes_penny_stocks_below_min_price() -> None:
    df = _panel()
    universe = build_investable_universe(df)
    assert universe.tolist() == [True, False, False, False, False]


def test_excludes_missing_price_mktcap_and_non_trading() -> None:
    df = _panel()
    universe = build_investable_universe(df)
    # row 2: missing price; row 3: zero mktcap; row 4: NT flag (no trading).
    assert not universe.iloc[2]
    assert not universe.iloc[3]
    assert not universe.iloc[4]


def test_financial_sic_exclusion_is_off_by_default() -> None:
    df = _panel()
    universe = build_investable_universe(df)
    # Row 4 fails on NT anyway; use a clean financial-SIC row to isolate the check.
    clean_financial = _panel(MthRetFlg=["CR", "CR", "CR", "CR", "CR"])
    universe = build_investable_universe(clean_financial)
    assert universe.iloc[4]  # financials included by default


def test_financial_sic_exclusion_when_enabled() -> None:
    df = _panel(MthRetFlg=["CR", "CR", "CR", "CR", "CR"])
    config = FeatureConfig(exclude_financials_from_universe=True)
    universe = build_investable_universe(df, config)
    assert not universe.iloc[4]
    assert universe.iloc[0]


def test_min_price_threshold_is_configurable() -> None:
    df = _panel()
    lenient = FeatureConfig(min_price_for_universe=0.1)
    universe = build_investable_universe(df, lenient)
    assert universe.iloc[1]  # $0.50 now passes with a lower threshold
