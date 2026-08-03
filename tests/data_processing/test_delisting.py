"""Tests for delisting-return compounding."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processing.delisting import apply_delisting_returns


@pytest.fixture
def result(crsp_monthly_df, crsp_delisting_df) -> pd.DataFrame:
    return apply_delisting_returns(crsp_monthly_df, crsp_delisting_df)


def _row(result: pd.DataFrame, permno: int) -> pd.Series:
    return result[result["PERMNO"] == permno].iloc[0]


def test_delisting_return_is_compounded_when_present(result) -> None:
    row = _row(result, 7)
    assert row["is_delisted"]
    assert not row["delisting_return_missing"]
    assert row["ret_adj"] == pytest.approx((1 + 0.05) * (1 + 0.10) - 1)


def test_missing_delisting_return_is_flagged_not_imputed(result) -> None:
    row = _row(result, 8)
    assert row["is_delisted"]
    assert row["delisting_return_missing"]
    # Left as the unadjusted MthRet, not a guessed proxy value.
    assert row["ret_adj"] == pytest.approx(0.02)


def test_null_monthly_return_treated_as_zero_for_compounding(result) -> None:
    # PERMNO 9's MthRet is null; ret_adj should reduce to the delisting
    # return alone, (1 + 0) * (1 + DelRet) - 1 == DelRet.
    row = _row(result, 9)
    assert row["is_delisted"]
    assert not row["delisting_return_missing"]
    assert row["ret_adj"] == pytest.approx(-0.50)


def test_never_delisted_security_is_unadjusted(result) -> None:
    row = _row(result, 10)
    assert not row["is_delisted"]
    assert not row["delisting_return_missing"]
    assert row["ret_adj"] == pytest.approx(0.03)


def test_only_last_observation_is_marked_delisted(result) -> None:
    # PERMNO 1 has 4 rows and is never delisted; none should be flagged, and
    # is_last_obs should be True for exactly its final (2021-06-30) row.
    permno_1 = result[result["PERMNO"] == 1].sort_values("MthCalDt")
    assert not permno_1["is_delisted"].any()
    assert permno_1["is_last_obs"].tolist() == [False, False, False, True]


def test_adjustment_does_not_affect_non_terminal_rows(crsp_monthly_df, crsp_delisting_df) -> None:
    # Add an earlier month for PERMNO 7 to confirm only the terminal row is touched.
    extra = crsp_monthly_df.iloc[[crsp_monthly_df.index[crsp_monthly_df["PERMNO"] == 7][0]]].copy()
    extra["MthCalDt"] = pd.Timestamp("2020-02-29")
    extra["MthRet"] = 0.5
    panel = pd.concat([crsp_monthly_df, extra], ignore_index=True)

    result = apply_delisting_returns(panel, crsp_delisting_df)
    earlier_row = result[(result["PERMNO"] == 7) & (result["MthCalDt"] == "2020-02-29")].iloc[0]
    assert not earlier_row["is_delisted"]
    assert earlier_row["ret_adj"] == pytest.approx(0.5)
