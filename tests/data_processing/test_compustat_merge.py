"""Tests for the point-in-time Compustat merge — the core look-ahead-bias guard.

Uses a hand-built PERMNO/MthCalDt/gvkey panel (rather than deriving `gvkey`
from ccm_linking) so each scenario's expected availability/expiry date is
unambiguous and independent of the CCM-linking tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_processing.compustat_merge import merge_compustat_point_in_time
from data_processing.config import MergeConfig


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": [1, 1, 1, 1, 2, 2, 2, 3],
            "MthCalDt": pd.to_datetime(
                [
                    "2020-04-30",
                    "2020-05-31",
                    "2020-06-30",
                    "2021-06-30",
                    "2016-06-30",
                    "2017-06-30",
                    "2017-07-31",
                    "2020-01-31",
                ]
            ),
            "gvkey": [100, 100, 100, 100, 200, 200, 200, np.nan],
        }
    )


def test_no_look_ahead_before_reporting_lag(compustat_df) -> None:
    # fy2019's datadate is 2019-12-31; with the default 6-month lag it isn't
    # available until 2020-06-30. Months before that must see NOTHING.
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    before_lag = result[result["MthCalDt"].isin(["2020-04-30", "2020-05-31"])]
    assert before_lag["cst_revt"].isna().all()
    assert before_lag["cst_available_date"].isna().all()


def test_data_available_exactly_at_reporting_lag_date(compustat_df) -> None:
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    row = result[result["MthCalDt"] == "2020-06-30"].iloc[0]
    assert row["cst_revt"] == 100.0
    assert row["cst_available_date"] == pd.Timestamp("2020-06-30")
    assert row["cst_age_months"] == 0
    assert not row["cst_expired"]


def test_newer_fiscal_year_supersedes_older_once_available(compustat_df) -> None:
    # By 2021-06-30, fy2020's data (also available exactly 2021-06-30) should
    # be used in preference to fy2019's — never the other way around.
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    row = result[result["MthCalDt"] == "2021-06-30"].iloc[0]
    assert row["cst_revt"] == 200.0


def test_data_expires_after_shelf_life(compustat_df) -> None:
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    at_12_months = result[result["MthCalDt"] == "2017-06-30"].iloc[0]
    past_12_months = result[result["MthCalDt"] == "2017-07-31"].iloc[0]

    assert at_12_months["cst_revt"] == 999.0
    assert not at_12_months["cst_expired"]

    assert pd.isna(past_12_months["cst_revt"])
    assert past_12_months["cst_expired"]
    # The availability date itself is retained for diagnostics even once expired.
    assert past_12_months["cst_available_date"] == pd.Timestamp("2016-06-30")


def test_configurable_shelf_life_changes_expiry_point(compustat_df) -> None:
    lenient_config = MergeConfig(max_fundamentals_age_months=24)
    result = merge_compustat_point_in_time(_panel(), compustat_df, lenient_config)
    row = result[result["MthCalDt"] == "2017-07-31"].iloc[0]
    assert row["cst_revt"] == 999.0
    assert not row["cst_expired"]


def test_row_without_gvkey_is_never_matched(compustat_df) -> None:
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    row = result[result["PERMNO"] == 3].iloc[0]
    assert pd.isna(row["cst_revt"])
    assert pd.isna(row["cst_available_date"])
    assert not row["cst_expired"]


def test_preserves_row_order_and_count(compustat_df) -> None:
    panel = _panel()
    result = merge_compustat_point_in_time(panel, compustat_df)
    assert len(result) == len(panel)
    assert result["PERMNO"].tolist() == panel["PERMNO"].tolist()
    assert result["MthCalDt"].tolist() == panel["MthCalDt"].tolist()


def test_dropped_columns_not_carried_into_panel(compustat_df) -> None:
    result = merge_compustat_point_in_time(_panel(), compustat_df)
    for col in ("cst_GVKEY", "cst_indfmt", "cst_consol", "cst_popsrc", "cst_datafmt", "cst_curcd"):
        assert col not in result.columns
