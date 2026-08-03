"""Tests for missingness flags and coverage reporting."""

from __future__ import annotations

import pandas as pd

from feature_engineering.missing import add_missingness_flags, coverage_report


def test_has_fundamentals_true_only_when_matched_and_not_expired() -> None:
    df = pd.DataFrame(
        {
            "cst_available_date": [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-01-01")],
            "cst_expired": [False, False, True],
        }
    )
    out = add_missingness_flags(df)
    assert out["has_fundamentals"].tolist() == [True, False, False]


def test_coverage_report_percentages() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0, None], "b": [1.0, 2.0, 3.0, 4.0]})
    report = coverage_report(df, ["a", "b"])
    assert report["a"] == 50.0
    assert report["b"] == 100.0
