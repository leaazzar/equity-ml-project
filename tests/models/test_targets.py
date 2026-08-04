from __future__ import annotations

import pandas as pd
import pytest

from equity_ml.models.config import ModelConfig
from equity_ml.models.targets import build_targets


def test_one_month_horizon_matches_next_month_return(simple_return_panel: pd.DataFrame) -> None:
    targets = build_targets(simple_return_panel, ModelConfig(horizon_months=1))
    permno1 = targets[targets["PERMNO"] == 1].sort_values("MthCalDt").reset_index(drop=True)
    # Every row except the last should have target_raw == 0.01 (next month's return).
    assert permno1["target_raw"].iloc[:-1].to_numpy() == pytest.approx(0.01)
    assert pd.isna(permno1["target_raw"].iloc[-1])
    assert not permno1["target_valid"].iloc[-1]


def test_multi_month_horizon_compounds_geometrically(simple_return_panel: pd.DataFrame) -> None:
    targets = build_targets(simple_return_panel, ModelConfig(horizon_months=3))
    permno1 = targets[targets["PERMNO"] == 1].sort_values("MthCalDt").reset_index(drop=True)
    expected = (1.01**3) - 1
    # First 33 rows have 3 full future months; last 3 do not.
    assert permno1["target_raw"].iloc[:-3].to_numpy() == pytest.approx(expected)
    assert permno1["target_raw"].iloc[-3:].isna().all()


def test_delisted_security_gets_nan_within_horizon_of_last_row(
    simple_return_panel: pd.DataFrame,
) -> None:
    targets = build_targets(simple_return_panel, ModelConfig(horizon_months=1))
    permno2 = targets[targets["PERMNO"] == 2].sort_values("MthCalDt").reset_index(drop=True)
    assert len(permno2) == 30
    assert permno2["target_raw"].iloc[:-1].to_numpy() == pytest.approx(0.02)
    assert pd.isna(permno2["target_raw"].iloc[-1])


def test_target_never_imputed_stays_nan(simple_return_panel: pd.DataFrame) -> None:
    targets = build_targets(simple_return_panel, ModelConfig(horizon_months=12))
    assert targets["target_raw"].isna().any()
    # NaN targets are never filled with 0 or any other placeholder.
    assert not (targets.loc[targets["target_valid"], "target_raw"] == 0).all()


def test_demeaned_target_removes_month_cross_sectional_mean(
    simple_return_panel: pd.DataFrame,
) -> None:
    targets = build_targets(simple_return_panel, ModelConfig(horizon_months=1))
    valid = targets[targets["target_valid"]]
    for _month, group in valid.groupby("MthCalDt"):
        assert group["target_demeaned"].sum() == pytest.approx(0.0, abs=1e-12)


def test_demeaning_excludes_non_investable_rows() -> None:
    months = pd.date_range("2000-01-31", periods=3, freq="ME")
    panel = pd.DataFrame(
        [
            {"PERMNO": 1, "MthCalDt": months[0], "ret_adj": 0.10, "is_investable": True},
            {"PERMNO": 2, "MthCalDt": months[0], "ret_adj": 0.90, "is_investable": False},
            {"PERMNO": 1, "MthCalDt": months[1], "ret_adj": 0.0, "is_investable": True},
            {"PERMNO": 2, "MthCalDt": months[1], "ret_adj": 0.0, "is_investable": False},
            {"PERMNO": 1, "MthCalDt": months[2], "ret_adj": 0.0, "is_investable": True},
            {"PERMNO": 2, "MthCalDt": months[2], "ret_adj": 0.0, "is_investable": False},
        ]
    )
    targets = build_targets(panel, ModelConfig(horizon_months=1))
    row = targets[(targets["PERMNO"] == 1) & (targets["MthCalDt"] == months[0])].iloc[0]
    # Only PERMNO 1 (investable) contributes to the month-0 mean, so its own
    # demeaned target is exactly 0 despite PERMNO 2's huge (non-investable) return.
    assert row["target_demeaned"] == pytest.approx(0.0)


def test_missing_required_column_raises() -> None:
    panel = pd.DataFrame({"PERMNO": [1], "MthCalDt": [pd.Timestamp("2020-01-31")]})
    with pytest.raises(KeyError):
        build_targets(panel, ModelConfig())
