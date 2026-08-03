"""Tests for the WRDS loader scaffolding.

These tests only verify the placeholder interface (NotImplementedError) and
never open a real WRDS connection, since no WRDS credentials are available
in this environment or in CI.
"""

from __future__ import annotations

import pytest

from equity_ml.data.wrds_loader import load_compustat_annual, load_crsp_monthly


def test_load_crsp_monthly_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_crsp_monthly()


def test_load_compustat_annual_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_compustat_annual()
