"""Synthetic panel fixtures for src/equity_ml/models/ tests. None of this is
real WRDS data — hand-constructed returns chosen so forward-return, split,
and portfolio math can be verified by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

MONTHS = pd.date_range("2000-01-31", periods=36, freq="ME")


@pytest.fixture
def simple_return_panel() -> pd.DataFrame:
    """Two PERMNOs over 36 months.

    - PERMNO 1: constant 1% return every month, always investable, full
      history (no delisting) -- used to hand-verify forward-return math.
    - PERMNO 2: constant 2% return, delists after month 30 (no rows after),
      always investable while present -- used to verify that a row within
      `horizon_months` of a security's last observation gets a NaN target.
    """
    rows = []
    for i, month in enumerate(MONTHS):
        rows.append(
            {
                "PERMNO": 1,
                "MthCalDt": month,
                "ret_adj": 0.01,
                "is_investable": True,
            }
        )
        if i < 30:
            rows.append(
                {
                    "PERMNO": 2,
                    "MthCalDt": month,
                    "ret_adj": 0.02,
                    "is_investable": True,
                }
            )
    return pd.DataFrame(rows)
