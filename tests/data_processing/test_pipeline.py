"""End-to-end pipeline test: interim Parquet files in -> master panel +
reports out, entirely on synthetic fixtures written to tmp_path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_processing.pipeline import run_pipeline


@pytest.fixture
def interim_dir(interim_frames, tmp_path: Path) -> Path:
    d = tmp_path / "interim"
    d.mkdir()
    for name, df in interim_frames.items():
        df.to_parquet(d / f"{name}.parquet", index=False)
    return d


def test_run_pipeline_end_to_end(interim_dir: Path, tmp_path: Path) -> None:
    result = run_pipeline(
        interim_dir=interim_dir,
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )

    assert result.output_path.exists()
    panel = pd.read_parquet(result.output_path)
    assert panel.duplicated(subset=["PERMNO", "MthCalDt"]).sum() == 0

    assert result.report_path.exists()
    assert (tmp_path / "reports" / "diagnostics.json").exists()
    assert (tmp_path / "reports" / "dedup_summary.json").exists()

    assert result.diagnostics.n_rows == len(panel)
    assert result.dedup_summary.n_rows_dropped == 1


def test_run_pipeline_missing_interim_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            interim_dir=tmp_path / "does_not_exist",
            processed_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )
