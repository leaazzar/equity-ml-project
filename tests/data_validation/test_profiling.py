"""Tests for generic dataset profiling: columns, dates, and duplicate keys."""

from __future__ import annotations

import pandas as pd

from data_validation.datasets import get_spec
from data_validation.profiling import (
    profile_columns,
    profile_dataset,
    profile_duplicates,
    suggest_primary_key,
)


def test_profile_columns_reports_nulls_and_uniques() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "y", "y", "z"]})
    profiles = {p.name: p for p in profile_columns(df)}
    assert profiles["a"].null_count == 1
    assert profiles["a"].null_pct == 25.0
    assert profiles["a"].n_unique == 2
    assert profiles["b"].null_count == 0
    assert profiles["b"].n_unique == 3


def test_suggest_primary_key_finds_smallest_unique_combo() -> None:
    # No single column is unique on its own (group and seq both repeat, val is
    # constant), but the pair (group, seq) together identifies every row.
    df = pd.DataFrame({"group": [1, 1, 2, 2], "seq": [1, 2, 1, 2], "val": ["a", "a", "a", "a"]})
    key = suggest_primary_key(df)
    assert key == ("group", "seq")


def test_suggest_primary_key_returns_none_when_rows_are_truly_identical(crsp_names_df) -> None:
    # crsp_names_df has one row exactly duplicated; no combination of columns
    # (up to the default search size) can distinguish the two copies.
    assert suggest_primary_key(crsp_names_df, max_combo_size=3) is None


def test_profile_duplicates_declared_key_distinguishes_exact_and_conflicting(
    crsp_monthly_df,
) -> None:
    spec = get_spec("crsp_monthly_stock")
    dup = profile_duplicates(crsp_monthly_df, spec)
    assert dup.key_source == "declared"
    assert dup.key_columns == ["PERMNO", "MthCalDt"]
    # PERMNO 2 (exact duplicate) + PERMNO 3 (conflicting duplicate) = 4 rows.
    assert dup.n_duplicate_key_rows == 4
    assert dup.n_excess_duplicate_rows == 2
    assert dup.n_exact_full_row_duplicates == 2
    assert dup.n_conflicting_duplicates == 2


def test_profile_dataset_end_to_end(crsp_monthly_df) -> None:
    spec = get_spec("crsp_monthly_stock")
    profile = profile_dataset(crsp_monthly_df, spec)
    assert profile.row_count == len(crsp_monthly_df)
    assert profile.column_count == crsp_monthly_df.shape[1]
    assert len(profile.date_columns) == 1
    date_profile = profile.date_columns[0]
    assert date_profile.column == "MthCalDt"
    assert date_profile.min_date == "2020-01-31"
    assert date_profile.max_date == "2020-02-29"
    assert date_profile.n_invalid == 0
