"""Tests for VALUE/QUALITY/INVESTMENT/LEVERAGE ratio features.

Covers zero/negative-denominator handling explicitly for every ratio, since
that's a named requirement, not just a nice-to-have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.ratios import (
    compute_investment_growth_features,
    compute_leverage_features,
    compute_quality_features,
    compute_value_features,
)


def _base_df() -> pd.DataFrame:
    # mktcap_millions is already unit-normalized ($ millions — see units.py);
    # these tests never use the raw CRSP MthCap column (which is $
    # thousands) directly, matching how compute_value_features is actually
    # called in the pipeline.
    return pd.DataFrame(
        {
            "mktcap_millions": [1000.0, 1000.0, 1000.0],
            "book_equity": [500.0, -100.0, 500.0],
            "enterprise_value": [1200.0, 1200.0, -50.0],
            "cst_ni": [50.0, 50.0, 50.0],
            "cst_oancf": [60.0, 60.0, 60.0],
            "cst_revt": [300.0, 300.0, 0.0],
            "cst_ebitda": [80.0, 80.0, 80.0],
        }
    )


def test_value_features_zero_and_negative_denominators_are_nan() -> None:
    df = _base_df()
    features = compute_value_features(df)

    # Row 1: negative book_equity -> bm is NaN (not a sign-flipped ratio).
    assert pd.isna(features["value_bm"].iloc[1])
    assert features["value_bm"].iloc[0] == pytest.approx(0.5)

    # Row 2: enterprise_value negative -> ebitda_to_ev and cf_to_ev NaN.
    assert pd.isna(features["value_ebitda_to_ev"].iloc[2])
    assert pd.isna(features["value_cf_to_ev"].iloc[2])

    # sales_to_price's denominator is mktcap_millions, not revt: revt == 0 in
    # the numerator is a valid (zero) sales-to-price ratio, not a missing one.
    assert features["value_sales_to_price"].iloc[2] == pytest.approx(0.0)


def test_value_features_no_infinite_values() -> None:
    df = _base_df()
    features = compute_value_features(df)
    for series in features.values():
        assert not np.isinf(series.astype("float64")).any()


def test_quality_features_roe_roa_denominators() -> None:
    df = pd.DataFrame(
        {
            "cst_ni": [10.0, 10.0, -5.0],
            "cst_seq": [100.0, 0.0, 50.0],
            "at_proxy": [200.0, 200.0, -10.0],
            "cst_gp": [40.0, 40.0, 40.0],
            "cst_revt": [300.0, 300.0, 300.0],
            "cst_cogs": [100.0, 100.0, 100.0],
            "cst_xsga": [50.0, None, 50.0],
            "cst_xint": [10.0, None, 10.0],
            "book_equity": [100.0, 100.0, 100.0],
            "cst_oancf": [30.0, 30.0, 30.0],
        }
    )
    features = compute_quality_features(df)

    assert features["quality_roe"].iloc[0] == pytest.approx(0.1)
    assert pd.isna(features["quality_roe"].iloc[1])  # seq == 0
    assert pd.isna(features["quality_roa"].iloc[2])  # at_proxy < 0


def test_quality_operating_profitability_treats_missing_cogs_xsga_xint_as_zero() -> None:
    df = pd.DataFrame(
        {
            "cst_ni": [10.0],
            "cst_seq": [100.0],
            "at_proxy": [200.0],
            "cst_gp": [40.0],
            "cst_revt": [300.0],
            "cst_cogs": [None],
            "cst_xsga": [None],
            "cst_xint": [None],
            "book_equity": [100.0],
            "cst_oancf": [30.0],
        }
    )
    features = compute_quality_features(df)
    # revt=300, cogs/xsga/xint treated as 0 -> op profit = 300 / 100 = 3.0
    assert features["quality_operating_profitability"].iloc[0] == pytest.approx(3.0)


def test_growth_features_pass_through_merged_columns() -> None:
    df = pd.DataFrame(
        {
            "cst_at_proxy_growth": [0.1],
            "cst_revt_growth": [0.2],
            "cst_capx_growth": [0.3],
            "cst_invt_growth": [0.4],
            "cst_rect_growth": [0.5],
            "cst_ppent_growth": [0.6],
            "cst_seq_growth": [0.7],
        }
    )
    features = compute_investment_growth_features(df)
    assert features["growth_asset"].iloc[0] == pytest.approx(0.1)
    assert features["growth_equity"].iloc[0] == pytest.approx(0.7)


def test_leverage_features_denominators() -> None:
    df = pd.DataFrame(
        {
            "cst_dlc": [10.0, 10.0],
            "cst_dltt": [40.0, 40.0],
            "total_debt": [50.0, 50.0],
            "net_debt": [30.0, 30.0],
            "at_proxy": [200.0, -5.0],
            "cst_seq": [100.0, 100.0],
            "cst_act": [80.0, 80.0],
            "cst_lct": [40.0, 0.0],
            "cst_che": [20.0, 20.0],
            "cst_ebit": [50.0, 50.0],
            "cst_xint": [5.0, 0.0],
        }
    )
    features = compute_leverage_features(df)

    assert features["leverage_debt_to_assets"].iloc[0] == pytest.approx(50.0 / 200.0)
    assert pd.isna(features["leverage_debt_to_assets"].iloc[1])  # at_proxy < 0
    assert pd.isna(features["leverage_current_ratio"].iloc[1])  # lct == 0
    assert pd.isna(features["leverage_interest_coverage"].iloc[1])  # xint == 0
    assert features["leverage_interest_coverage"].iloc[0] == pytest.approx(10.0)


def test_leverage_features_no_infinite_values() -> None:
    df = pd.DataFrame(
        {
            "cst_dlc": [10.0],
            "cst_dltt": [40.0],
            "total_debt": [50.0],
            "net_debt": [30.0],
            "at_proxy": [0.0],
            "cst_seq": [0.0],
            "cst_act": [80.0],
            "cst_lct": [0.0],
            "cst_che": [20.0],
            "cst_ebit": [50.0],
            "cst_xint": [0.0],
        }
    )
    features = compute_leverage_features(df)
    for series in features.values():
        assert not np.isinf(series.astype("float64")).any()
