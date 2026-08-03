"""Missingness flags and coverage summaries.

No feature is ever imputed (see every module's docstrings and
FEATURE_DICTIONARY.md) — missing source data always produces a missing
feature value. What this module adds is a small number of *flags* that make
the reason for missingness explicit and economically meaningful, rather than
requiring a user to infer it from which columns happen to be null.
"""

from __future__ import annotations

import pandas as pd


def add_missingness_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `has_fundamentals`: whether this row has usable (non-expired,
    point-in-time-available) Compustat data at all.

    `link_matched` (whether a CCM link was active) already exists on the
    master panel from `data_processing` and is not duplicated here. Together,
    `link_matched` and `has_fundamentals` explain *why* every `cst_*`-derived
    and accounting-ratio feature is missing for a given row, without needing
    a separate flag per individual ratio.
    """
    out = panel.copy()
    out["has_fundamentals"] = out["cst_available_date"].notna() & ~out["cst_expired"]
    return out


def coverage_report(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Percentage of non-null values per column, overall."""
    n = len(df)
    return pd.Series(
        {col: round(100 * df[col].notna().sum() / n, 4) if n else 0.0 for col in columns},
        name="pct_non_null",
    )
