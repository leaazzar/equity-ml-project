"""Tests for loading the interim Parquet inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_processing.io import REQUIRED_DATASETS, load_all_interim, load_interim


@pytest.fixture
def interim_dir(interim_frames, tmp_path: Path) -> Path:
    d = tmp_path / "interim"
    d.mkdir()
    for name, df in interim_frames.items():
        df.to_parquet(d / f"{name}.parquet", index=False)
    return d


def test_load_interim_reads_a_single_dataset(interim_dir: Path) -> None:
    df = load_interim("crsp_monthly_stock", interim_dir)
    assert isinstance(df, pd.DataFrame)
    assert "PERMNO" in df.columns


def test_load_interim_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_interim("crsp_monthly_stock", tmp_path)


def test_load_all_interim_returns_every_required_dataset(interim_dir: Path) -> None:
    frames = load_all_interim(interim_dir)
    assert set(frames) == set(REQUIRED_DATASETS)


def test_load_all_interim_excludes_crsp_names() -> None:
    assert "crsp_names" not in REQUIRED_DATASETS
