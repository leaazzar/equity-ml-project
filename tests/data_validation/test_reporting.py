"""Tests for report generation (Markdown + JSON) under reports/data_validation/."""

from __future__ import annotations

import json
from pathlib import Path

from data_validation.cleaning import clean_and_write
from data_validation.datasets import get_spec
from data_validation.integrity import check_delisting_missing_returns
from data_validation.profiling import profile_dataset
from data_validation.reporting import write_reports


def test_write_reports_creates_expected_files(
    crsp_monthly_df, crsp_delisting_df, tmp_path: Path
) -> None:
    spec = get_spec("crsp_monthly_stock")
    profile = profile_dataset(crsp_monthly_df, spec)
    finding = check_delisting_missing_returns(crsp_delisting_df)
    _, cleaning_summary = clean_and_write(crsp_monthly_df, spec, tmp_path / "interim")

    reports_dir = tmp_path / "reports"
    summary_path = write_reports(
        profiles={"crsp_monthly_stock": profile},
        findings=[finding],
        cleaning_summaries=[cleaning_summary],
        reports_dir=reports_dir,
    )

    assert summary_path == reports_dir / "summary.md"
    assert summary_path.exists()
    assert (reports_dir / "crsp_monthly_stock_profile.json").exists()
    assert (reports_dir / "integrity_checks.json").exists()
    assert (reports_dir / "cleaning_summary.json").exists()

    profile_json = json.loads((reports_dir / "crsp_monthly_stock_profile.json").read_text())
    assert profile_json["row_count"] == len(crsp_monthly_df)

    summary_text = summary_path.read_text()
    assert "crsp_monthly_stock" in summary_text
    assert "delisting_missing_delret" in summary_text
