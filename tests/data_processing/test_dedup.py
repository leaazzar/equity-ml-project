"""Tests for conflicting-duplicate-key resolution."""

from __future__ import annotations

from data_processing.dedup import resolve_duplicate_keys


def test_resolve_duplicate_keys_keeps_most_complete_row(crsp_monthly_df) -> None:
    resolved, summary = resolve_duplicate_keys(crsp_monthly_df, ["PERMNO", "MthCalDt"])

    assert summary.rows_in == len(crsp_monthly_df)
    assert summary.n_duplicate_groups == 1
    assert summary.n_rows_dropped == 1
    assert summary.rows_out == len(crsp_monthly_df) - 1

    permno_6_rows = resolved[resolved["PERMNO"] == 6]
    assert len(permno_6_rows) == 1
    assert permno_6_rows.iloc[0]["Ticker"] == "FFF"
    assert permno_6_rows.iloc[0]["CUSIP"] == "FFF"


def test_resolve_duplicate_keys_leaves_unique_keys_untouched(crsp_monthly_df) -> None:
    resolved, _ = resolve_duplicate_keys(crsp_monthly_df, ["PERMNO", "MthCalDt"])
    assert resolved["PERMNO"].isin([1]).sum() == 4


def test_resolve_duplicate_keys_no_duplicates_is_a_no_op() -> None:
    import pandas as pd

    df = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": pd.to_datetime(["2020-01-31", "2020-02-29"])})
    resolved, summary = resolve_duplicate_keys(df, ["PERMNO", "MthCalDt"])
    assert summary.n_rows_dropped == 0
    assert len(resolved) == 2
