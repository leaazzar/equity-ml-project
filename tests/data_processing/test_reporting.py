"""Tests for merge-diagnostics report generation."""

from __future__ import annotations

import json
from pathlib import Path

from data_processing.diagnostics import compute_diagnostics
from data_processing.panel import build_master_panel
from data_processing.reporting import write_reports


def test_write_reports_creates_expected_files(interim_frames, tmp_path: Path) -> None:
    panel, dedup_summary = build_master_panel(interim_frames)
    diagnostics = compute_diagnostics(panel)

    reports_dir = tmp_path / "reports"
    diagnostics_path = write_reports(diagnostics, dedup_summary, reports_dir)

    assert diagnostics_path == reports_dir / "diagnostics.md"
    assert diagnostics_path.exists()
    assert (reports_dir / "diagnostics.json").exists()
    assert (reports_dir / "dedup_summary.json").exists()

    payload = json.loads((reports_dir / "diagnostics.json").read_text())
    assert payload["n_rows"] == diagnostics.n_rows

    text = diagnostics_path.read_text()
    assert "Firm coverage over time" in text
    assert "Missing accounting variables" in text
