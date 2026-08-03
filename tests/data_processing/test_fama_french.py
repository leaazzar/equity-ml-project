"""Tests for the Fama-French month-key merge."""

from __future__ import annotations

from data_processing.fama_french import merge_fama_french


def test_merge_fama_french_matches_by_month(crsp_monthly_df, fama_french_df) -> None:
    result = merge_fama_french(crsp_monthly_df, fama_french_df)
    assert result["ff_mktrf"].notna().all()
    assert "ff_smb" in result.columns and "ff_umd" in result.columns


def test_merge_fama_french_preserves_row_count(crsp_monthly_df, fama_french_df) -> None:
    result = merge_fama_french(crsp_monthly_df, fama_french_df)
    assert len(result) == len(crsp_monthly_df)


def test_merge_fama_french_leaves_null_for_unmatched_month(crsp_monthly_df, fama_french_df) -> None:
    ff_missing_month = fama_french_df[fama_french_df["dateff"] != "2020-04-30"]
    result = merge_fama_french(crsp_monthly_df, ff_missing_month)
    unmatched = result[result["MthCalDt"] == "2020-04-30"]
    assert unmatched["ff_mktrf"].isna().all()


def test_merge_fama_french_column_values_are_correct(crsp_monthly_df, fama_french_df) -> None:
    ff = fama_french_df.copy()
    ff.loc[ff["dateff"] == "2020-06-30", "mktrf"] = 0.99
    result = merge_fama_french(crsp_monthly_df, ff)
    row = result[result["MthCalDt"] == "2020-06-30"].iloc[0]
    assert row["ff_mktrf"] == 0.99
