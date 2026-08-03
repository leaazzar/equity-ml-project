"""Automated validation: leakage, rolling-window/key integrity, denominator
handling, cross-sectional isolation, and reproducibility.

Two kinds of checks live here:

- **Structural checks** (duplicates, calendar gaps, infinite values, registry
  consistency, denominator handling, return bounds) run cheaply against the
  full panel every time the pipeline runs.
- **Behavioral checks** (no-look-ahead via truncation, cross-sectional
  isolation, reproducibility) require re-running feature computation and are
  run on a small, deterministic sample (a handful of PERMNOs/months) to keep
  pipeline runtime reasonable — they are exercised exhaustively against
  synthetic data in `tests/feature_engineering/test_validation.py`, where
  runtime cost isn't a concern.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering.registry import FEATURE_REGISTRY

RETURN_BASED_FEATURES = (
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_9m",
    "mom_12m",
    "mom_12_1",
    "mom_6_1",
    "reversal_1m",
    "max_ret_12m",
    "min_ret_12m",
)

# feature -> raw denominator column(s) that must be strictly positive for a
# non-null value to be valid. Used by check_denominator_handling.
_POSITIVE_DENOMINATOR_FEATURES: dict[str, str] = {
    "quality_roe": "cst_seq",
    "quality_roa": "at_proxy",
    "quality_gross_profitability": "at_proxy",
    "quality_net_profit_margin": "cst_revt",
    "quality_gross_margin": "cst_revt",
    "quality_asset_turnover": "at_proxy",
    "quality_cf_profitability": "at_proxy",
    "quality_cf_margin": "cst_revt",
    "quality_accruals": "at_proxy",
    "leverage_debt_to_assets": "at_proxy",
    "leverage_debt_to_equity": "cst_seq",
    "leverage_lt_debt_ratio": "at_proxy",
    "leverage_current_ratio": "cst_lct",
    "leverage_cash_ratio": "cst_lct",
    "leverage_net_debt_to_assets": "at_proxy",
    "leverage_interest_coverage": "cst_xint",
    "liquidity_share_turnover": "ShrOut",
}


def _mismatch_mask(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    """True wherever `a` and `b` disagree — including when one is NaN and the
    other isn't. Plain `(a - b).abs() > tol` silently treats "NaN vs. a real
    number" as *not* a mismatch (NaN comparisons are always False), which
    would let exactly the kind of leakage this module exists to catch slip
    through undetected.
    """
    a = a.astype("float64")
    b = b.astype("float64")
    both_present = a.notna() & b.notna()
    presence_differs = a.notna() != b.notna()
    value_differs = both_present & ((a - b).abs() > tol)
    return presence_differs | value_differs


@dataclass
class ValidationResult:
    """Outcome of one validation check."""

    check_name: str
    passed: bool
    detail: str
    n_affected: int = 0


def check_no_duplicate_keys(df: pd.DataFrame) -> ValidationResult:
    n_dup = int(df.duplicated(subset=["PERMNO", "MthCalDt"]).sum())
    return ValidationResult(
        "no_duplicate_permno_month_keys", n_dup == 0, f"{n_dup} duplicate PERMNO-month rows", n_dup
    )


def check_no_calendar_gaps(df: pd.DataFrame) -> ValidationResult:
    """Every PERMNO's observations should be calendar-month-contiguous.

    Rolling-window features assume this (see time_series_features.py's
    module docstring) — a gap would mean a fixed-size row-position window no
    longer corresponds to the intended calendar-month window for that
    PERMNO.
    """
    month = df.sort_values(["PERMNO", "MthCalDt"])["MthCalDt"].dt.to_period("M")
    permno = df.sort_values(["PERMNO", "MthCalDt"])["PERMNO"]
    diffs = month.groupby(permno).diff()
    gap_count = int(((diffs.dropna()).apply(lambda d: d.n) > 1).sum())
    return ValidationResult(
        "no_calendar_gaps",
        gap_count == 0,
        f"{gap_count} PERMNO-month transitions skip a calendar month",
        gap_count,
    )


def check_no_infinite_values(df: pd.DataFrame, feature_cols: list[str]) -> ValidationResult:
    n_inf = 0
    bad_cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        count = int(np.isinf(df[col].astype("float64")).sum())
        if count:
            n_inf += count
            bad_cols.append(col)
    return ValidationResult(
        "no_infinite_values",
        n_inf == 0,
        f"{n_inf} infinite values across columns {bad_cols}" if n_inf else "no infinite values",
        n_inf,
    )


def check_registry_source_columns_exist(available_columns: set[str]) -> ValidationResult:
    """Every registry-declared source column must actually exist somewhere
    in the pipeline's working data (raw panel columns or derived/intermediate
    columns computed en route — see accounting.py)."""
    missing: dict[str, list[str]] = {}
    for spec in FEATURE_REGISTRY:
        absent = [c for c in spec.source_columns if c not in available_columns]
        if absent:
            missing[spec.name] = absent
    return ValidationResult(
        "registry_source_columns_exist",
        len(missing) == 0,
        f"missing source columns: {missing}" if missing else "all declared source columns exist",
        len(missing),
    )


def check_return_based_feature_bounds(df: pd.DataFrame) -> ValidationResult:
    """Cumulative/monthly returns can never be below -100% (a total loss)."""
    n_bad = 0
    bad_cols = []
    for col in RETURN_BASED_FEATURES:
        if col not in df.columns:
            continue
        count = int((df[col] < -1.0).sum())
        if count:
            n_bad += count
            bad_cols.append(col)
    return ValidationResult(
        "return_based_features_above_negative_100pct",
        n_bad == 0,
        f"{n_bad} values below -100% in {bad_cols}"
        if n_bad
        else "all return-based features >= -100%",
        n_bad,
    )


def check_denominator_handling(df: pd.DataFrame) -> ValidationResult:
    """For every ratio requiring a strictly positive denominator, confirm no
    non-null feature value exists where that raw denominator was <= 0."""
    violations: dict[str, int] = {}
    for feature, denom_col in _POSITIVE_DENOMINATOR_FEATURES.items():
        if feature not in df.columns or denom_col not in df.columns:
            continue
        bad = df[feature].notna() & (df[denom_col] <= 0)
        count = int(bad.sum())
        if count:
            violations[feature] = count
    n_total = sum(violations.values())
    return ValidationResult(
        "denominator_handling",
        n_total == 0,
        f"non-null values with a non-positive denominator: {violations}" if violations else "ok",
        n_total,
    )


def check_no_leakage_via_truncation(
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_master_panel: pd.DataFrame,
    cutoff_dates: list[pd.Timestamp],
    feature_cols: list[str],
    sample_permnos: list[int] | None = None,
) -> ValidationResult:
    """For each cutoff date, rebuild features from data truncated at that
    date and confirm the cutoff month's rows are identical to building from
    the full sample. This is the definitive no-look-ahead check: if any
    feature used data beyond its observation month, truncating the input
    would change that month's computed value.
    """
    full_result = build_fn(raw_master_panel)
    mismatches: dict[str, int] = {}

    for cutoff in cutoff_dates:
        truncated_input = raw_master_panel[raw_master_panel["MthCalDt"] <= cutoff]
        if sample_permnos is not None:
            truncated_input = truncated_input[truncated_input["PERMNO"].isin(sample_permnos)]
        truncated_result = build_fn(truncated_input)

        full_slice = full_result[full_result["MthCalDt"] == cutoff]
        trunc_slice = truncated_result[truncated_result["MthCalDt"] == cutoff]
        if sample_permnos is not None:
            full_slice = full_slice[full_slice["PERMNO"].isin(sample_permnos)]

        merged = full_slice[["PERMNO", *feature_cols]].merge(
            trunc_slice[["PERMNO", *feature_cols]],
            on="PERMNO",
            suffixes=("_full", "_trunc"),
            how="inner",
        )
        for col in feature_cols:
            if f"{col}_full" not in merged or f"{col}_trunc" not in merged:
                continue
            mismatch = _mismatch_mask(merged[f"{col}_full"], merged[f"{col}_trunc"], tol=1e-8)
            n_mismatch = int(mismatch.sum())
            if n_mismatch:
                mismatches[f"{cutoff.date()}::{col}"] = n_mismatch

    n_total = sum(mismatches.values())
    return ValidationResult(
        "no_leakage_via_truncation",
        n_total == 0,
        f"mismatches (full-sample vs. truncated-input) at: {mismatches}" if mismatches else "ok",
        n_total,
    )


def check_reproducibility(
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw_master_panel: pd.DataFrame,
    feature_cols: list[str],
) -> ValidationResult:
    """Running the same build twice on the same input must produce identical output."""
    first = build_fn(raw_master_panel)
    second = build_fn(raw_master_panel)
    mismatches = {}
    for col in feature_cols:
        if col not in first.columns or col not in second.columns:
            continue
        mismatch = _mismatch_mask(first[col], second[col], tol=1e-12)
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            mismatches[col] = n_mismatch
    n_total = sum(mismatches.values())
    return ValidationResult(
        "reproducibility",
        n_total == 0,
        f"non-reproducible columns: {mismatches}" if mismatches else "ok",
        n_total,
    )


def check_cross_sectional_isolation(
    transform_fn: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    month_col: pd.Series,
    sample_months: list[pd.Period],
) -> ValidationResult:
    """A month's model-ready transform must not change when other months are removed."""
    full_result = transform_fn(df)
    mismatches: dict[str, int] = {}

    for month in sample_months:
        subset = df[month_col == month]
        subset_result = transform_fn(subset)
        full_subset_result = full_result.loc[subset.index]

        mismatch = _mismatch_mask(subset_result, full_subset_result, tol=1e-8)
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            mismatches[str(month)] = n_mismatch

    n_total = sum(mismatches.values())
    return ValidationResult(
        "cross_sectional_isolation",
        n_total == 0,
        f"months affected by other months' data: {mismatches}" if mismatches else "ok",
        n_total,
    )
