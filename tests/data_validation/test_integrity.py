"""Tests for cross-dataset referential integrity checks.

Each test uses the small synthetic fixtures in conftest.py, which embed
exactly one deliberate issue per check so expected counts are unambiguous.
"""

from __future__ import annotations

from data_validation import integrity


def test_delisting_permno_in_crsp(crsp_monthly_df, crsp_delisting_df) -> None:
    finding = integrity.check_delisting_permno_in_crsp(crsp_monthly_df, crsp_delisting_df)
    assert finding.n_affected == 1
    assert finding.status == "warning"
    assert "999" in finding.sample


def test_delisting_missing_returns(crsp_delisting_df) -> None:
    finding = integrity.check_delisting_missing_returns(crsp_delisting_df)
    assert finding.n_affected == 1
    assert finding.n_total == 3


def test_delisting_successor_permno_coverage_ok_when_no_successors(
    crsp_monthly_df, crsp_delisting_df
) -> None:
    finding = integrity.check_delisting_successor_permno_coverage(
        crsp_monthly_df, crsp_delisting_df
    )
    assert finding.n_affected == 0
    assert finding.status == "ok"


def test_possible_missing_delisting_records(crsp_monthly_df, crsp_delisting_df) -> None:
    # months_buffer=0: any PERMNO whose last observation is before the overall max
    # date, and absent from crsp_delisting, is a candidate. PERMNO 3 and 4 qualify;
    # PERMNO 2's last obs is also before the max but it does have a delisting record.
    finding = integrity.check_possible_missing_delisting_records(
        crsp_monthly_df, crsp_delisting_df, months_buffer=0
    )
    assert finding.n_affected == 2
    assert set(finding.sample) == {"3", "4"}


def test_ccm_permno_coverage_explains_pre_window_gap(ccm_link_table_df, crsp_monthly_df) -> None:
    finding = integrity.check_ccm_permno_coverage(ccm_link_table_df, crsp_monthly_df)
    # LPERMNO 999's link ends in 1999, before crsp_monthly_df's earliest date
    # (2020-01-31), so it's an explained gap, not flagged as unexplained.
    assert finding.n_affected == 0
    assert finding.status == "ok"


def test_ccm_gvkey_coverage_explains_pre_window_gap(ccm_link_table_df, compustat_df) -> None:
    finding = integrity.check_ccm_gvkey_coverage(ccm_link_table_df, compustat_df)
    assert finding.n_affected == 0
    assert finding.status == "ok"


def test_compustat_gvkey_in_ccm_ok_when_fully_covered(ccm_link_table_df, compustat_df) -> None:
    finding = integrity.check_compustat_gvkey_in_ccm(ccm_link_table_df, compustat_df)
    assert finding.n_affected == 0


def test_ccm_link_period_overlaps_detects_overlapping_window(ccm_link_table_df) -> None:
    finding = integrity.check_ccm_link_period_overlaps(ccm_link_table_df)
    assert finding.n_affected == 1
    assert finding.status == "warning"
    assert finding.sample == ["gvkey=30, LIID=01"]


def test_ccm_link_period_overlaps_ignores_adjacent_non_overlapping_windows(
    ccm_link_table_df,
) -> None:
    finding = integrity.check_ccm_link_period_overlaps(ccm_link_table_df)
    assert "gvkey=10, LIID=01" not in finding.sample


def test_fama_french_date_coverage_detects_missing_month(crsp_monthly_df, fama_french_df) -> None:
    finding = integrity.check_fama_french_date_coverage(fama_french_df, crsp_monthly_df)
    assert finding.n_affected == 1
    assert finding.status == "warning"
    assert finding.sample == ["2020-02"]


def test_identifier_naming_consistency_flags_permno_permco_gvkey() -> None:
    finding = integrity.check_identifier_naming_consistency()
    assert finding.n_affected == 3
    assert finding.status == "info"
