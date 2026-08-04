from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_ml.backtest.config import BacktestConfig
from equity_ml.backtest.performance import (
    compute_performance_metrics,
    cost_sensitivity_table,
    factor_regression,
)


def _portfolio_returns(
    returns: list[float], turnover: float = 0.0, model: str = "ridge"
) -> pd.DataFrame:
    months = pd.date_range("2020-01-31", periods=len(returns), freq="ME")
    return pd.DataFrame(
        {
            "MthCalDt": months,
            "model": model,
            "gross_return": returns,
            "turnover": [turnover] * len(returns),
            "cost": [0.0] * len(returns),
            "net_return": returns,
        }
    )


def test_compute_performance_metrics_positive_constant_return_series() -> None:
    # Constant positive return -> std is 0 -> Sharpe/Sortino undefined (NaN), but
    # annualized return and hit rate are well-defined.
    portfolio_returns = _portfolio_returns([0.01] * 24)
    metrics = compute_performance_metrics(portfolio_returns, BacktestConfig())
    gross = metrics[(metrics["model"] == "ridge") & (metrics["kind"] == "gross")].iloc[0]
    assert gross["annualized_return"] == pytest.approx(0.01 * 12)
    assert gross["hit_rate"] == pytest.approx(1.0)
    assert gross["max_drawdown"] == pytest.approx(0.0)


def test_max_drawdown_on_known_path() -> None:
    # +10%, -20%, +5%: cumulative = 1.10, 0.88, 0.924. Drawdown trough = 0.88/1.10 - 1 = -0.20.
    portfolio_returns = _portfolio_returns([0.10, -0.20, 0.05])
    metrics = compute_performance_metrics(portfolio_returns, BacktestConfig())
    gross = metrics[metrics["kind"] == "gross"].iloc[0]
    assert gross["max_drawdown"] == pytest.approx(-0.20, abs=1e-6)


def test_sortino_only_penalizes_downside() -> None:
    # All positive returns with variance -> Sortino should be NaN (no downside observations).
    portfolio_returns = _portfolio_returns([0.01, 0.02, 0.03, 0.01, 0.02])
    metrics = compute_performance_metrics(portfolio_returns, BacktestConfig())
    gross = metrics[metrics["kind"] == "gross"].iloc[0]
    assert np.isnan(gross["sortino"])


def test_cost_sensitivity_table_has_one_row_per_bps_level() -> None:
    portfolio_returns = _portfolio_returns([0.01, -0.01, 0.02, -0.02] * 6, turnover=1.0)
    config = BacktestConfig(cost_sensitivity_bps=(0.0, 10.0, 50.0))
    table = cost_sensitivity_table(portfolio_returns, config)
    assert len(table) == 3
    # Higher transaction costs should never increase annualized return when turnover > 0.
    returns_by_bps = table.set_index("cost_bps")["annualized_return"]
    assert returns_by_bps[50.0] < returns_by_bps[10.0] < returns_by_bps[0.0]


def test_factor_regression_recovers_known_alpha_and_beta() -> None:
    rng = np.random.default_rng(0)
    n = 60
    months = pd.date_range("2015-01-31", periods=n, freq="ME")
    mktrf = rng.normal(scale=0.04, size=n)
    true_alpha = 0.002
    true_beta = 0.5
    net_return = true_alpha + true_beta * mktrf  # noise-free, exactly recoverable

    portfolio_returns = pd.DataFrame(
        {
            "MthCalDt": months,
            "model": "ridge",
            "gross_return": net_return,
            "turnover": 0.0,
            "cost": 0.0,
            "net_return": net_return,
        }
    )
    factor_panel = pd.DataFrame({"MthCalDt": months, "ff_mktrf": mktrf})
    config = BacktestConfig(factor_columns=("ff_mktrf",))
    result = factor_regression(portfolio_returns, factor_panel, config)

    row = result.iloc[0]
    assert row["alpha"] == pytest.approx(true_alpha, abs=1e-8)
    assert row["ff_mktrf_beta"] == pytest.approx(true_beta, abs=1e-8)
    assert row["n_months"] == n


def test_factor_regression_skips_model_with_too_few_overlapping_months() -> None:
    months = pd.date_range("2015-01-31", periods=3, freq="ME")
    portfolio_returns = pd.DataFrame(
        {
            "MthCalDt": months,
            "model": "ridge",
            "gross_return": [0.01, 0.02, 0.01],
            "turnover": 0.0,
            "cost": 0.0,
            "net_return": [0.01, 0.02, 0.01],
        }
    )
    factor_panel = pd.DataFrame({"MthCalDt": months, "ff_mktrf": [0.01, 0.02, 0.01]})
    result = factor_regression(
        portfolio_returns, factor_panel, BacktestConfig(factor_columns=("ff_mktrf",))
    )
    assert result.empty
