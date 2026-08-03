"""Tests for interim-output cleaning: exact-duplicate removal and date casting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_validation.cleaning import clean_and_write
from data_validation.datasets import get_spec


def test_clean_and_write_drops_only_exact_duplicates(crsp_monthly_df, tmp_path: Path) -> None:
    spec = get_spec("crsp_monthly_stock")
    cleaned, summary = clean_and_write(crsp_monthly_df, spec, tmp_path)

    assert summary.rows_in == len(crsp_monthly_df)
    # Only PERMNO 2's exact duplicate is dropped; PERMNO 3's conflicting
    # duplicate-key rows (differing CUSIP/Ticker) are left untouched.
    assert summary.n_exact_duplicates_dropped == 1
    assert summary.rows_out == len(crsp_monthly_df) - 1
    assert len(cleaned) == summary.rows_out


def test_clean_and_write_casts_date_columns(crsp_monthly_df, tmp_path: Path) -> None:
    spec = get_spec("crsp_monthly_stock")
    cleaned, _ = clean_and_write(crsp_monthly_df, spec, tmp_path)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["MthCalDt"])


def test_clean_and_write_writes_readable_parquet(crsp_monthly_df, tmp_path: Path) -> None:
    spec = get_spec("crsp_monthly_stock")
    _, summary = clean_and_write(crsp_monthly_df, spec, tmp_path)
    output_path = Path(summary.output_path)
    assert output_path.exists()
    roundtrip = pd.read_parquet(output_path)
    assert len(roundtrip) == summary.rows_out


def test_clean_and_write_does_not_mutate_input(crsp_monthly_df, tmp_path: Path) -> None:
    spec = get_spec("crsp_monthly_stock")
    original_len = len(crsp_monthly_df)
    original_dtype = crsp_monthly_df["MthCalDt"].dtype
    clean_and_write(crsp_monthly_df, spec, tmp_path)
    assert len(crsp_monthly_df) == original_len
    assert crsp_monthly_df["MthCalDt"].dtype == original_dtype
