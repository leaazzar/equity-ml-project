"""Tests for merge-diagnostics computation."""

from __future__ import annotations

from data_processing.diagnostics import compute_diagnostics
from data_processing.panel import build_master_panel


def test_compute_diagnostics_no_duplicates_after_dedup(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    assert diagnostics.n_duplicate_permno_months == 0
    assert diagnostics.n_unique_permno_months == diagnostics.n_rows


def test_compute_diagnostics_gvkey_coverage(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    n_with_gvkey = panel["gvkey"].notna().sum()
    assert diagnostics.pct_gvkey_matched == round(100 * n_with_gvkey / len(panel), 4)


def test_compute_diagnostics_delisting_counts(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    assert diagnostics.n_delisted_rows == 3  # PERMNOs 7, 8, 9
    assert diagnostics.n_delisted_missing_return == 1  # PERMNO 8


def test_compute_diagnostics_accounting_var_missing_pct_has_all_columns(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    assert "cst_revt" in diagnostics.accounting_var_missing_pct
    assert 0.0 <= diagnostics.accounting_var_missing_pct["cst_revt"] <= 100.0


def test_compute_diagnostics_firm_coverage_by_year_spans_expected_years(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    years = {row["year"] for row in diagnostics.firm_coverage_by_year}
    assert {2016, 2017, 2018, 2020, 2021, 2025}.issubset(years)


def test_compute_diagnostics_firm_coverage_counts_are_consistent(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)
    for row in diagnostics.firm_coverage_by_year:
        assert row["n_permno_with_gvkey"] <= row["n_permno"]
        assert row["n_permno_with_compustat"] <= row["n_permno_with_gvkey"]
