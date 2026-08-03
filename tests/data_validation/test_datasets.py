"""Tests for the dataset registry."""

from __future__ import annotations

import pytest

from data_validation.datasets import DATASETS, get_spec

REQUIRED_FILENAMES = {
    "crsp_monthly_stock.csv",
    "crsp_delisting.csv",
    "crsp_names.csv",
    "ccm_link_table.csv",
    "compustat_fundamentals_annual.csv",
    "fama_french_5f_momentum_monthly.csv",
}


def test_registry_covers_all_required_datasets() -> None:
    filenames = {spec.filename for spec in DATASETS.values()}
    assert filenames == REQUIRED_FILENAMES


def test_get_spec_returns_known_dataset() -> None:
    spec = get_spec("crsp_monthly_stock")
    assert spec.filename == "crsp_monthly_stock.csv"
    assert spec.declared_key == ("PERMNO", "MthCalDt")


def test_get_spec_raises_for_unknown_dataset() -> None:
    with pytest.raises(KeyError):
        get_spec("not_a_real_dataset")


def test_crsp_names_has_no_declared_key() -> None:
    # crsp_names has no validity-period date columns, so no row-level primary
    # key can be declared with confidence — it must be auto-detected instead.
    assert get_spec("crsp_names").declared_key is None
