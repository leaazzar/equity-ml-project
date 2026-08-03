"""Tests for raw CSV loading and date parsing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_validation.datasets import get_spec
from data_validation.io import load_raw, parse_date_column


def test_load_raw_reads_all_rows(raw_dir: Path) -> None:
    df = load_raw(get_spec("crsp_monthly_stock"), raw_dir)
    assert len(df) == 2
    assert list(df.columns) == [
        "PERMNO",
        "HdrCUSIP",
        "CUSIP",
        "Ticker",
        "PERMCO",
        "SICCD",
        "MthCalDt",
        "MthPrc",
        "MthCap",
        "MthRet",
        "MthRetx",
        "MthRetFlg",
        "MthVol",
        "ShrOut",
    ]


def test_load_raw_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(get_spec("crsp_monthly_stock"), tmp_path)


def test_parse_date_column_handles_valid_dates() -> None:
    spec = get_spec("crsp_monthly_stock")
    raw = pd.Series(["2020-01-31", "2020-02-29"])
    result = parse_date_column(raw, "MthCalDt", spec)
    assert result.n_valid == 2
    assert result.n_invalid == 0
    assert result.n_sentinel == 0


def test_parse_date_column_recognizes_sentinel() -> None:
    spec = get_spec("ccm_link_table")
    raw = pd.Series(["2020-01-31", "E"])
    result = parse_date_column(raw, "LINKENDDT", spec)
    assert result.n_valid == 1
    assert result.n_sentinel == 1
    assert result.n_invalid == 0


def test_parse_date_column_flags_invalid_values() -> None:
    spec = get_spec("crsp_monthly_stock")
    raw = pd.Series(["2020-01-31", "not-a-date", None])
    result = parse_date_column(raw, "MthCalDt", spec)
    assert result.n_valid == 1
    assert result.n_invalid == 1
    assert "not-a-date" in result.invalid_samples
