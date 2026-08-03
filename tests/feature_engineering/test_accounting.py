"""Tests for derived accounting quantities and point-in-time YoY growth rates."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processing.config import MergeConfig
from feature_engineering.accounting import (
    add_derived_accounting_columns,
    compute_growth_compustat_table,
    merge_growth_features,
)
from feature_engineering.config import FeatureConfig


def test_at_proxy_is_liabilities_plus_equity() -> None:
    df = pd.DataFrame(
        {
            "cst_lt": [100.0],
            "cst_seq": [50.0],
            "cst_pstk": [None],
            "cst_txditc": [None],
            "cst_dlc": [0.0],
            "cst_dltt": [0.0],
            "cst_che": [0.0],
            "mktcap_millions": [0.0],
        }
    )
    out = add_derived_accounting_columns(df)
    assert out["at_proxy"].iloc[0] == pytest.approx(150.0)


def test_book_equity_treats_missing_pstk_and_txditc_as_zero() -> None:
    df = pd.DataFrame(
        {
            "cst_lt": [0.0],
            "cst_seq": [50.0],
            "cst_pstk": [None],
            "cst_txditc": [None],
            "cst_dlc": [0.0],
            "cst_dltt": [0.0],
            "cst_che": [0.0],
            "mktcap_millions": [0.0],
        }
    )
    out = add_derived_accounting_columns(df)
    assert out["book_equity"].iloc[0] == pytest.approx(50.0)


def test_book_equity_subtracts_preferred_and_adds_txditc() -> None:
    df = pd.DataFrame(
        {
            "cst_lt": [0.0],
            "cst_seq": [50.0],
            "cst_pstk": [10.0],
            "cst_txditc": [5.0],
            "cst_dlc": [0.0],
            "cst_dltt": [0.0],
            "cst_che": [0.0],
            "mktcap_millions": [0.0],
        }
    )
    out = add_derived_accounting_columns(df)
    assert out["book_equity"].iloc[0] == pytest.approx(45.0)


def test_enterprise_value_and_net_debt() -> None:
    # mktcap_millions=1000 represents a $1,000 million ($1B) market cap —
    # already unit-normalized (see units.py); NOT the raw CRSP MthCap, which
    # would be in $ thousands and require dividing by 1,000 first.
    df = pd.DataFrame(
        {
            "cst_lt": [0.0],
            "cst_seq": [0.0],
            "cst_pstk": [10.0],
            "cst_txditc": [0.0],
            "cst_dlc": [20.0],
            "cst_dltt": [30.0],
            "cst_che": [15.0],
            "mktcap_millions": [1000.0],
        }
    )
    out = add_derived_accounting_columns(df)
    assert out["total_debt"].iloc[0] == pytest.approx(50.0)
    assert out["net_debt"].iloc[0] == pytest.approx(35.0)
    # EV = mktcap + total_debt + pstk - cash = 1000 + 50 + 10 - 15 = 1045 ($ millions)
    assert out["enterprise_value"].iloc[0] == pytest.approx(1045.0)


def _compustat_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GVKEY": [1, 1, 1, 2],
            "datadate": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31", "2019-12-31"]),
            "lt": [100.0, 110.0, 0.0, 50.0],
            "seq": [
                50.0,
                55.0,
                60.0,
                -10.0,
            ],  # gvkey 1 fy2020 at_proxy = 0+60=60 (prior fy2019=165, positive)
            "revt": [200.0, 220.0, 0.0, 100.0],
            "capx": [10.0, 0.0, 5.0, 5.0],
            "invt": [30.0, 33.0, 36.0, 10.0],
            "rect": [20.0, 22.0, 24.0, 8.0],
            "ppent": [40.0, 44.0, 48.0, 15.0],
        }
    )


def test_growth_requires_positive_prior_value() -> None:
    growth = compute_growth_compustat_table(_compustat_fixture())
    gvkey1 = growth[growth["GVKEY"] == 1].sort_values("datadate").reset_index(drop=True)

    # fy2018 has no prior -> NaN.
    assert pd.isna(gvkey1.loc[0, "revt_growth"])
    # fy2019 vs fy2018: revt growth = (220-200)/200 = 0.10
    assert gvkey1.loc[1, "revt_growth"] == pytest.approx(0.10)
    # fy2020 revt = 0, prior (fy2019) = 220 > 0 -> growth = (0-220)/220 = -1.0 (valid, not NaN)
    assert gvkey1.loc[2, "revt_growth"] == pytest.approx(-1.0)
    # fy2020's capx prior (fy2019) = 0 -> NaN (non-positive base)
    assert pd.isna(gvkey1.loc[2, "capx_growth"])


def test_growth_does_not_cross_gvkeys() -> None:
    growth = compute_growth_compustat_table(_compustat_fixture())
    gvkey2 = growth[growth["GVKEY"] == 2]
    # gvkey 2 has only one fiscal year -> no prior available -> NaN, regardless
    # of gvkey 1 having a fy2019 row (must not be picked up as "prior").
    assert pd.isna(gvkey2["revt_growth"].iloc[0])


def test_merge_growth_features_respects_point_in_time_lag() -> None:
    panel = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "MthCalDt": pd.to_datetime(["2019-05-31", "2019-06-30", "2020-06-30"]),
            "gvkey": [1, 1, 1],
        }
    )
    merged = merge_growth_features(panel, _compustat_fixture(), FeatureConfig(), MergeConfig())
    # 2019-05-31: fy2018 isn't available until 2019-06-30 (6mo lag) -> no match at all yet.
    assert pd.isna(merged.loc[0, "cst_revt_growth"])
    # 2020-06-30: fy2019 (available exactly 2020-06-30) is matched, and its
    # growth rate (0.10, vs. fy2018) is what should be attached here.
    assert merged.loc[2, "cst_revt_growth"] == pytest.approx(0.10)


def test_merge_growth_features_drops_redundant_bookkeeping_columns() -> None:
    panel = pd.DataFrame({"PERMNO": [1], "MthCalDt": pd.to_datetime(["2020-06-30"]), "gvkey": [1]})
    merged = merge_growth_features(panel, _compustat_fixture(), FeatureConfig(), MergeConfig())
    assert "cst_available_date" not in merged.columns
    assert "cst_age_months" not in merged.columns
    assert "cst_expired" not in merged.columns
