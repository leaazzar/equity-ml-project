"""Tests for month-by-month winsorization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.config import FeatureConfig
from feature_engineering.outliers import winsorize_by_month


def test_winsorize_clips_to_month_specific_bounds() -> None:
    # 100 investable observations in January (uniform 1..100), one extreme
    # outlier. 1st/99th percentile bounds should clip it.
    values = pd.Series([*list(range(1, 101)), 10000.0])
    month = pd.Series(["2020-01"] * 101)
    is_investable = pd.Series([True] * 101)
    config = FeatureConfig(winsorize_lower_pct=0.01, winsorize_upper_pct=0.99)

    result = winsorize_by_month(values, month, is_investable, config)
    assert result.iloc[-1] < 10_000.0
    assert result.iloc[-1] == pytest.approx(values.quantile(0.99))


def test_winsorize_thresholds_are_isolated_by_month() -> None:
    jan = list(range(1, 21))  # small values
    feb = list(range(1000, 1020))  # unrelated, much larger values
    values = pd.Series(jan + feb, dtype="float64")
    month = pd.Series(["2020-01"] * 20 + ["2020-02"] * 20)
    is_investable = pd.Series([True] * 40)
    config = FeatureConfig(min_cross_section_for_winsorize=5)

    result = winsorize_by_month(values, month, is_investable, config)
    # February's values must not be clipped down to January's tiny range.
    assert result.iloc[20:].min() >= 1000


def test_winsorize_uses_only_investable_universe_for_thresholds() -> None:
    values = pd.Series([*list(range(1, 21)), 999.0], dtype="float64")
    month = pd.Series(["2020-01"] * 21)
    # The extreme value (999) is flagged non-investable, so it must not pull
    # the computed percentile bounds upward.
    is_investable = pd.Series([True] * 20 + [False])
    config = FeatureConfig(min_cross_section_for_winsorize=5)

    result = winsorize_by_month(values, month, is_investable, config)
    upper_bound = pd.Series(range(1, 21), dtype="float64").quantile(config.winsorize_upper_pct)
    assert result.iloc[-1] == pytest.approx(upper_bound)


def test_winsorize_skips_months_with_too_few_observations() -> None:
    values = pd.Series([1.0, 2.0, 1000.0])
    month = pd.Series(["2020-01"] * 3)
    is_investable = pd.Series([True] * 3)
    config = FeatureConfig(min_cross_section_for_winsorize=10)  # more than available

    result = winsorize_by_month(values, month, is_investable, config)
    # Too few observations to winsorize reliably -> pass through unchanged.
    pd.testing.assert_series_equal(result, values)


def test_winsorize_never_produces_infinite_values() -> None:
    values = pd.Series([1.0, np.inf, -np.inf, 5.0, 10.0] * 5)
    month = pd.Series(["2020-01"] * 25)
    is_investable = pd.Series([True] * 25)
    result = winsorize_by_month(values, month, is_investable)
    # inf/-inf were already in the input; winsorize shouldn't introduce new
    # ones for the finite values, though pre-existing inf inputs are a
    # separate upstream concern (features must never produce inf themselves).
    finite_mask = np.isfinite(values)
    assert not np.isinf(result[finite_mask]).any()
