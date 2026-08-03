"""Tests for the validation suite — including deliberately broken
implementations to prove the leakage/reproducibility/isolation checks
actually catch what they claim to."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.validation import (
    check_cross_sectional_isolation,
    check_denominator_handling,
    check_no_calendar_gaps,
    check_no_duplicate_keys,
    check_no_infinite_values,
    check_no_leakage_via_truncation,
    check_registry_source_columns_exist,
    check_reproducibility,
    check_return_based_feature_bounds,
)


def test_check_no_duplicate_keys_detects_duplicates() -> None:
    df = pd.DataFrame({"PERMNO": [1, 1, 2], "MthCalDt": pd.to_datetime(["2020-01-31"] * 3)})
    result = check_no_duplicate_keys(df)
    assert not result.passed
    assert result.n_affected > 0


def test_check_no_duplicate_keys_passes_when_unique() -> None:
    df = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": pd.to_datetime(["2020-01-31"] * 2)})
    assert check_no_duplicate_keys(df).passed


def test_check_no_calendar_gaps_detects_a_skipped_month() -> None:
    df = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "MthCalDt": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"]),
        }
    )
    result = check_no_calendar_gaps(df)
    assert not result.passed
    assert result.n_affected == 1


def test_check_no_infinite_values_detects_inf() -> None:
    df = pd.DataFrame({"feat_a": [1.0, np.inf, 3.0]})
    result = check_no_infinite_values(df, ["feat_a"])
    assert not result.passed
    assert result.n_affected == 1


def test_check_registry_source_columns_exist_detects_missing_columns() -> None:
    # An essentially empty set of available columns -> every registry entry's
    # source columns should be reported missing.
    result = check_registry_source_columns_exist(set())
    assert not result.passed
    assert result.n_affected > 0


def test_check_return_based_feature_bounds_detects_impossible_return() -> None:
    df = pd.DataFrame({"mom_1m": [-1.5, 0.1]})
    result = check_return_based_feature_bounds(df)
    assert not result.passed
    assert result.n_affected == 1


def test_check_denominator_handling_detects_violation() -> None:
    df = pd.DataFrame({"quality_roe": [0.1, 0.2], "cst_seq": [100.0, -5.0]})
    result = check_denominator_handling(df)
    assert not result.passed
    assert result.n_affected == 1


def test_check_denominator_handling_passes_when_clean() -> None:
    df = pd.DataFrame({"quality_roe": [0.1, np.nan], "cst_seq": [100.0, -5.0]})
    result = check_denominator_handling(df)
    assert result.passed


def _leakage_free_build(panel: pd.DataFrame) -> pd.DataFrame:
    """A correct, backward-looking-only 'feature': rolling sum of the trailing 2 rows."""
    out = panel.sort_values(["PERMNO", "MthCalDt"]).copy()
    out["feat"] = (
        out.groupby("PERMNO")["value"]
        .rolling(2, min_periods=2)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return out


def _leaky_build(panel: pd.DataFrame) -> pd.DataFrame:
    """A deliberately broken 'feature' that uses the NEXT row's value (look-ahead)."""
    out = panel.sort_values(["PERMNO", "MthCalDt"]).copy()
    out["feat"] = out.groupby("PERMNO")["value"].shift(-1)  # future value!
    return out


def _synthetic_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": [1, 1, 1, 1, 1],
            "MthCalDt": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
            ),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_no_leakage_check_passes_for_backward_looking_feature() -> None:
    panel = _synthetic_panel()
    cutoffs = [pd.Timestamp("2020-03-31")]
    result = check_no_leakage_via_truncation(_leakage_free_build, panel, cutoffs, ["feat"])
    assert result.passed


def test_no_leakage_check_fails_for_forward_looking_feature() -> None:
    panel = _synthetic_panel()
    cutoffs = [pd.Timestamp("2020-03-31")]
    result = check_no_leakage_via_truncation(_leaky_build, panel, cutoffs, ["feat"])
    assert not result.passed
    assert result.n_affected > 0


def test_reproducibility_check_passes_for_deterministic_build() -> None:
    panel = _synthetic_panel()
    result = check_reproducibility(_leakage_free_build, panel, ["feat"])
    assert result.passed


def test_reproducibility_check_fails_for_nondeterministic_build() -> None:
    call_count = {"n": 0}

    def _nondeterministic_build(panel: pd.DataFrame) -> pd.DataFrame:
        call_count["n"] += 1
        out = panel.copy()
        out["feat"] = out["value"] + (0.0 if call_count["n"] == 1 else 1.0)
        return out

    panel = _synthetic_panel()
    result = check_reproducibility(_nondeterministic_build, panel, ["feat"])
    assert not result.passed


def test_cross_sectional_isolation_passes_for_isolated_transform() -> None:
    df = pd.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            "is_investable": [True] * 6,
        }
    )
    month = pd.Series(["2020-01"] * 3 + ["2020-02"] * 3)

    def transform_fn(subset: pd.DataFrame) -> pd.Series:
        sub_month = month.loc[subset.index]
        return subset.groupby(sub_month)["val"].rank(pct=True)

    result = check_cross_sectional_isolation(transform_fn, df, month, ["2020-01", "2020-02"])
    assert result.passed


def test_cross_sectional_isolation_fails_for_leaky_transform() -> None:
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]})
    month = pd.Series(["2020-01"] * 3 + ["2020-02"] * 3)

    def leaky_transform_fn(subset: pd.DataFrame) -> pd.Series:
        # Deliberately does NOT group by month — a real "isolation" bug
        # would compute a rank/percentile across whatever rows happen to be
        # passed in, so a February value's rank differs depending on
        # whether January's (much smaller) values are also present.
        return subset["val"].rank(pct=True)

    result = check_cross_sectional_isolation(leaky_transform_fn, df, month, ["2020-01", "2020-02"])
    assert not result.passed
