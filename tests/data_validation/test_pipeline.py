"""End-to-end pipeline test using tiny synthetic raw CSVs (see conftest.raw_dir).

This never touches the real (large, gitignored) data/raw/ directory, so it
runs the same way in CI as it does locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_validation.datasets import DATASETS
from data_validation.pipeline import run_pipeline


def test_run_pipeline_end_to_end(raw_dir: Path, tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"

    result = run_pipeline(raw_dir=raw_dir, interim_dir=interim_dir, reports_dir=reports_dir)

    assert set(result.profiles) == set(DATASETS)
    assert len(result.cleaning_summaries) == len(DATASETS)
    assert result.report_path.exists()

    for name in DATASETS:
        parquet_path = interim_dir / f"{name}.parquet"
        assert parquet_path.exists()
        # Every interim file should be readable and non-empty.
        assert len(pd.read_parquet(parquet_path)) > 0

    assert (reports_dir / "summary.md").exists()
    assert (reports_dir / "integrity_checks.json").exists()


def test_run_pipeline_leaves_raw_files_untouched(raw_dir: Path, tmp_path: Path) -> None:
    before = {p.name: p.read_text() for p in raw_dir.glob("*.csv")}
    run_pipeline(
        raw_dir=raw_dir, interim_dir=tmp_path / "interim", reports_dir=tmp_path / "reports"
    )
    after = {p.name: p.read_text() for p in raw_dir.glob("*.csv")}
    assert before == after


def test_run_pipeline_missing_raw_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            raw_dir=tmp_path / "does_not_exist",
            interim_dir=tmp_path / "interim",
            reports_dir=tmp_path / "reports",
        )
