from __future__ import annotations

import pandas as pd
import pytest

from equity_ml.backtest.config import BacktestConfig
from equity_ml.backtest.engine import compute_portfolio_returns


def test_gross_return_is_weighted_sum_of_realized_returns() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {"PERMNO": [1, 2], "MthCalDt": date, "model": "ridge", "weight": [1.0, -1.0]}
    )
    realized = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": date, "target_raw": [0.10, 0.02]})
    result = compute_portfolio_returns(weights, realized, BacktestConfig(transaction_cost_bps=0.0))
    row = result.iloc[0]
    assert row["gross_return"] == pytest.approx(1.0 * 0.10 + -1.0 * 0.02)
    assert row["net_return"] == pytest.approx(row["gross_return"])  # zero cost


def test_first_rebalance_turnover_is_half_sum_abs_weights() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {"PERMNO": [1, 2], "MthCalDt": date, "model": "ridge", "weight": [1.0, -1.0]}
    )
    realized = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": date, "target_raw": [0.0, 0.0]})
    result = compute_portfolio_returns(weights, realized, BacktestConfig())
    assert result.iloc[0]["turnover"] == pytest.approx(0.5 * (1.0 + 1.0))


def test_unchanged_weights_across_periods_have_zero_turnover_after_first() -> None:
    jan = pd.Timestamp("2020-01-31")
    feb = pd.Timestamp("2020-02-29")
    weights = pd.DataFrame(
        {
            "PERMNO": [1, 2, 1, 2],
            "MthCalDt": [jan, jan, feb, feb],
            "model": "ridge",
            "weight": [1.0, -1.0, 1.0, -1.0],
        }
    )
    realized = pd.DataFrame(
        {
            "PERMNO": [1, 2, 1, 2],
            "MthCalDt": [jan, jan, feb, feb],
            "target_raw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = compute_portfolio_returns(weights, realized, BacktestConfig())
    feb_row = result[result["MthCalDt"] == feb].iloc[0]
    assert feb_row["turnover"] == pytest.approx(0.0)


def test_full_turnover_when_portfolio_completely_changes() -> None:
    jan = pd.Timestamp("2020-01-31")
    feb = pd.Timestamp("2020-02-29")
    weights = pd.DataFrame(
        {
            "PERMNO": [1, 2, 3, 4],
            "MthCalDt": [jan, jan, feb, feb],
            "model": "ridge",
            "weight": [1.0, -1.0, 1.0, -1.0],
        }
    )
    realized = pd.DataFrame(
        {
            "PERMNO": [1, 2, 3, 4],
            "MthCalDt": [jan, jan, feb, feb],
            "target_raw": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = compute_portfolio_returns(weights, realized, BacktestConfig())
    feb_row = result[result["MthCalDt"] == feb].iloc[0]
    # Entirely new names in Feb -> turnover = 0.5 * (|1-0|+|-1-0|+|0-1|+|0-(-1)|) = 2.0
    assert feb_row["turnover"] == pytest.approx(2.0)


def test_net_return_subtracts_cost_proportional_to_turnover() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {"PERMNO": [1, 2], "MthCalDt": date, "model": "ridge", "weight": [1.0, -1.0]}
    )
    realized = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": date, "target_raw": [0.0, 0.0]})
    config = BacktestConfig(transaction_cost_bps=100.0)  # 1%
    result = compute_portfolio_returns(weights, realized, config)
    row = result.iloc[0]
    expected_cost = row["turnover"] * (100.0 / 10_000.0)
    assert row["cost"] == pytest.approx(expected_cost)
    assert row["net_return"] == pytest.approx(row["gross_return"] - expected_cost)


def test_position_with_missing_realized_return_is_dropped_not_zero_filled() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {"PERMNO": [1, 2], "MthCalDt": date, "model": "ridge", "weight": [1.0, -1.0]}
    )
    realized = pd.DataFrame({"PERMNO": [1], "MthCalDt": date, "target_raw": [0.05]})
    result = compute_portfolio_returns(weights, realized, BacktestConfig(transaction_cost_bps=0.0))
    # Only PERMNO 1's contribution counts; PERMNO 2 is silently absent, not treated as 0 return.
    assert result.iloc[0]["gross_return"] == pytest.approx(0.05)


def test_models_are_scored_independently() -> None:
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame(
        {
            "PERMNO": [1, 2, 1, 2],
            "MthCalDt": date,
            "model": ["ridge", "ridge", "lasso", "lasso"],
            "weight": [1.0, -1.0, 0.5, -0.5],
        }
    )
    realized = pd.DataFrame({"PERMNO": [1, 2], "MthCalDt": date, "target_raw": [0.1, 0.0]})
    result = compute_portfolio_returns(weights, realized, BacktestConfig(transaction_cost_bps=0.0))
    ridge_return = result[result["model"] == "ridge"]["gross_return"].iloc[0]
    lasso_return = result[result["model"] == "lasso"]["gross_return"].iloc[0]
    assert ridge_return == pytest.approx(0.1)
    assert lasso_return == pytest.approx(0.05)
