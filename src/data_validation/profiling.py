"""Generic dataset profiling: schema, dtypes, missingness, duplicates, dates.

Everything in this module operates on whatever columns are actually present
in a `pandas.DataFrame` — it does not hardcode column lists. The only
dataset-specific input is which key to check for duplicates (from
`DatasetSpec.declared_key`, or auto-detected here when absent).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from data_validation.datasets import DatasetSpec
from data_validation.io import DateParseResult, parse_date_column

MAX_SAMPLES = 5
MAX_AUTO_KEY_SIZE = 3


@dataclass
class ColumnProfile:
    """Generic, dataset-agnostic statistics for a single column."""

    name: str
    dtype: str
    row_count: int
    null_count: int
    null_pct: float
    n_unique: int
    sample_values: list[str] = field(default_factory=list)


@dataclass
class DateColumnProfile:
    """Statistics for a column that was parsed as a date."""

    column: str
    min_date: str | None
    max_date: str | None
    n_valid: int
    n_sentinel: int
    n_invalid: int
    invalid_samples: list[str] = field(default_factory=list)

    @classmethod
    def from_parse_result(cls, result: DateParseResult) -> DateColumnProfile:
        valid = result.parsed.dropna()
        return cls(
            column=result.column,
            min_date=str(valid.min().date()) if not valid.empty else None,
            max_date=str(valid.max().date()) if not valid.empty else None,
            n_valid=result.n_valid,
            n_sentinel=result.n_sentinel,
            n_invalid=result.n_invalid,
            invalid_samples=result.invalid_samples,
        )


@dataclass
class DuplicateKeyProfile:
    """Duplicate-key statistics for a dataset's (declared or detected) primary key."""

    key_columns: list[str]
    key_source: str  # "declared" or "auto-detected" or "none-found"
    n_rows: int
    n_duplicate_key_rows: int
    n_excess_duplicate_rows: int
    n_exact_full_row_duplicates: int
    n_conflicting_duplicates: int


@dataclass
class DatasetProfile:
    """Full profile of one dataset: schema, columns, dates, duplicates."""

    name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    date_columns: list[DateColumnProfile]
    duplicates: DuplicateKeyProfile


def _sample_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique()[:MAX_SAMPLES]
    return list(values)


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    """Compute generic per-column statistics for every column in `df`."""
    n = len(df)
    profiles = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                row_count=n,
                null_count=null_count,
                null_pct=round(100 * null_count / n, 4) if n else 0.0,
                n_unique=int(series.nunique(dropna=True)),
                sample_values=_sample_values(series),
            )
        )
    return profiles


def suggest_primary_key(
    df: pd.DataFrame, max_combo_size: int = MAX_AUTO_KEY_SIZE
) -> tuple[str, ...] | None:
    """Search for the smallest column combination that uniquely identifies each row.

    Tries single columns, then pairs, then triples (in that order), returning the
    first combination found with zero duplicate keys. Returns `None` if nothing up
    to `max_combo_size` columns achieves uniqueness (e.g. the file contains
    genuine duplicate records with no distinguishing column at all).
    """
    n = len(df)
    if n == 0:
        return ()
    for size in range(1, max_combo_size + 1):
        for combo in itertools.combinations(df.columns, size):
            if df.duplicated(subset=list(combo)).sum() == 0:
                return combo
    return None


def profile_duplicates(df: pd.DataFrame, spec: DatasetSpec) -> DuplicateKeyProfile:
    """Compute duplicate-key statistics using the declared key, or auto-detect one."""
    if spec.declared_key is not None:
        key_columns = list(spec.declared_key)
        key_source = "declared"
    else:
        detected = suggest_primary_key(df)
        key_columns = list(detected) if detected else list(df.columns)
        key_source = "auto-detected" if detected else "none-found"

    dup_mask = df.duplicated(subset=key_columns, keep=False)
    n_duplicate_key_rows = int(dup_mask.sum())
    n_excess_duplicate_rows = int(df.duplicated(subset=key_columns, keep="first").sum())

    if n_duplicate_key_rows:
        dup_rows = df.loc[dup_mask]
        n_exact_full_row = int(dup_rows.duplicated(keep=False).sum())
    else:
        n_exact_full_row = 0

    return DuplicateKeyProfile(
        key_columns=key_columns,
        key_source=key_source,
        n_rows=len(df),
        n_duplicate_key_rows=n_duplicate_key_rows,
        n_excess_duplicate_rows=n_excess_duplicate_rows,
        n_exact_full_row_duplicates=n_exact_full_row,
        n_conflicting_duplicates=n_duplicate_key_rows - n_exact_full_row,
    )


def profile_dataset(df: pd.DataFrame, spec: DatasetSpec) -> DatasetProfile:
    """Build a full profile of `df` (row/column counts, columns, dates, duplicates)."""
    date_profiles = [
        DateColumnProfile.from_parse_result(parse_date_column(df[col], col, spec))
        for col in spec.date_columns
    ]
    return DatasetProfile(
        name=spec.name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=profile_columns(df),
        date_columns=date_profiles,
        duplicates=profile_duplicates(df, spec),
    )
