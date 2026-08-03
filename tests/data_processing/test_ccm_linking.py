"""Tests for point-in-time CRSP PERMNO -> gvkey resolution via CCM.

These are the most safety-critical tests in this package: getting the
interval-join logic wrong would silently attach the wrong firm's accounting
data (or none at all) to a security-month.
"""

from __future__ import annotations

import pandas as pd

from data_processing.ccm_linking import resolve_gvkey_for_panel
from data_processing.config import MergeConfig


def _gvkey_for(resolved: pd.DataFrame, permno: int) -> list:
    return resolved.loc[resolved["PERMNO"] == permno, "gvkey"].tolist()


def test_resolve_gvkey_matches_open_ended_link(crsp_monthly_df, ccm_link_table_df) -> None:
    resolved = resolve_gvkey_for_panel(crsp_monthly_df, ccm_link_table_df)
    # PERMNO 1's link (gvkey 100) is open-ended from 2019 — every one of its
    # rows (spanning 2020-2021) must resolve, including the one a year later.
    assert _gvkey_for(resolved, 1) == [100, 100, 100, 100]
    assert resolved.loc[resolved["PERMNO"] == 1, "link_matched"].all()


def test_resolve_gvkey_open_ended_link_matches_far_future_date(
    crsp_monthly_df, ccm_link_table_df
) -> None:
    # PERMNO 4's link starts 2018-01-01, open-ended; its CRSP row is 2025-01-31,
    # seven years later. This is the regression test for the bug where the
    # open-ended sentinel wasn't applied to already-parsed (NaT) LINKENDDT
    # values coming from the interim Parquet files.
    resolved = resolve_gvkey_for_panel(crsp_monthly_df, ccm_link_table_df)
    row = resolved[resolved["PERMNO"] == 4].iloc[0]
    assert row["gvkey"] == 400
    assert row["link_matched"]


def test_resolve_gvkey_no_link_at_all_is_unmatched(crsp_monthly_df, ccm_link_table_df) -> None:
    resolved = resolve_gvkey_for_panel(crsp_monthly_df, ccm_link_table_df)
    row = resolved[resolved["PERMNO"] == 3].iloc[0]
    assert pd.isna(row["gvkey"])
    assert not row["link_matched"]


def test_resolve_gvkey_expired_link_is_unmatched_after_end_date(
    crsp_monthly_df, ccm_link_table_df
) -> None:
    # PERMNO 5's link closes 2016-12-31; its CRSP row is 2018-01-31 — after
    # the link ended, so it must NOT still resolve to gvkey 500.
    resolved = resolve_gvkey_for_panel(crsp_monthly_df, ccm_link_table_df)
    row = resolved[resolved["PERMNO"] == 5].iloc[0]
    assert pd.isna(row["gvkey"])
    assert not row["link_matched"]


def test_resolve_gvkey_preserves_input_row_order(crsp_monthly_df, ccm_link_table_df) -> None:
    resolved = resolve_gvkey_for_panel(crsp_monthly_df, ccm_link_table_df)
    assert resolved["PERMNO"].tolist() == crsp_monthly_df["PERMNO"].tolist()
    assert resolved["MthCalDt"].tolist() == crsp_monthly_df["MthCalDt"].tolist()


def test_resolve_gvkey_before_link_start_is_unmatched() -> None:
    crsp = pd.DataFrame({"PERMNO": [1], "MthCalDt": pd.to_datetime(["2005-01-31"])})
    ccm = pd.DataFrame(
        {
            "gvkey": [100],
            "LINKPRIM": ["P"],
            "LPERMNO": [1],
            "LINKDT": pd.to_datetime(["2010-01-01"]),
            "LINKENDDT": pd.to_datetime([None]),
        }
    )
    resolved = resolve_gvkey_for_panel(crsp, ccm)
    assert pd.isna(resolved.iloc[0]["gvkey"])
    assert not resolved.iloc[0]["link_matched"]


def test_resolve_gvkey_tie_break_prefers_configured_priority() -> None:
    # Two simultaneously-valid links for the same PERMNO on the exact same
    # LINKDT (not observed in real data, but exercised here directly): the
    # non-primary link is listed first, so this also checks that the tie
    # break isn't just "whichever row happens to sort first".
    crsp = pd.DataFrame({"PERMNO": [1], "MthCalDt": pd.to_datetime(["2020-06-30"])})
    ccm = pd.DataFrame(
        {
            "gvkey": [200, 100],
            "LINKPRIM": ["C", "P"],
            "LPERMNO": [1, 1],
            "LINKDT": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "LINKENDDT": pd.to_datetime([None, None]),
        }
    )
    config = MergeConfig(linkprim_priority=("P", "C", "J", "N"))
    resolved = resolve_gvkey_for_panel(crsp, ccm, config)
    assert resolved.iloc[0]["gvkey"] == 100
