"""Tests for the model-ready transform: rank-scale, z-score, and the
combined winsorize+rank pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.config import FeatureConfig
from feature_engineering.transforms import (
    build_model_ready_column,
    rank_transform_by_month,
    zscore_transform_by_month,
)


def test_rank_transform_bounds_and_ordering() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    month = pd.Series(["2020-01"] * 5)
    is_investable = pd.Series([True] * 5)

    result = rank_transform_by_month(values, month, is_investable)
    assert result.min() >= -1
    assert result.max() <= 1
    assert result.is_monotonic_increasing


def test_rank_transform_non_investable_rows_are_nan() -> None:
    values = pd.Series([10.0, 20.0, 30.0])
    month = pd.Series(["2020-01"] * 3)
    is_investable = pd.Series([True, True, False])

    result = rank_transform_by_month(values, month, is_investable)
    assert pd.isna(result.iloc[2])
    assert not pd.isna(result.iloc[0])


def test_rank_transform_missing_raw_value_stays_missing() -> None:
    values = pd.Series([10.0, None, 30.0])
    month = pd.Series(["2020-01"] * 3)
    is_investable = pd.Series([True, True, True])

    result = rank_transform_by_month(values, month, is_investable)
    assert pd.isna(result.iloc[1])


def test_rank_transform_isolated_by_month() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    month = pd.Series(["2020-01"] * 3 + ["2020-02"] * 3)
    is_investable = pd.Series([True] * 6)

    result = rank_transform_by_month(values, month, is_investable)
    # Both months' middle observation should rank at the same relative
    # position (the middle of 3), regardless of the other month's scale.
    assert result.iloc[1] == pytest.approx(result.iloc[4])


def test_zscore_transform_mean_zero_within_month() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    month = pd.Series(["2020-01"] * 5)
    is_investable = pd.Series([True] * 5)

    result = zscore_transform_by_month(values, month, is_investable)
    assert result.mean() == pytest.approx(0.0, abs=1e-9)


def test_zscore_transform_zero_variance_is_nan_not_inf() -> None:
    values = pd.Series([5.0, 5.0, 5.0])
    month = pd.Series(["2020-01"] * 3)
    is_investable = pd.Series([True] * 3)

    result = zscore_transform_by_month(values, month, is_investable)
    assert result.isna().all()


def test_build_model_ready_column_winsorizes_then_ranks() -> None:
    values = pd.Series([*list(range(1, 101)), 1000000.0], dtype="float64")
    month = pd.Series(["2020-01"] * 101)
    is_investable = pd.Series([True] * 101)
    config = FeatureConfig(winsorize_lower_pct=0.01, winsorize_upper_pct=0.99)

    result = build_model_ready_column(values, month, is_investable, config)
    # Winsorization clips the extreme outlier down to the 99th-percentile
    # value, so it now ties with whichever original point(s) sat at or above
    # that boundary — sharing the same (high, but not uniquely maximal) rank.
    # That tie is the whole point: without winsorizing first, the outlier
    # would dominate the ranking on its own.
    assert result.max() <= 1.0
    n_at_max = (result == result.max()).sum()
    assert n_at_max > 1, "the winsorized outlier should tie with other boundary values"
