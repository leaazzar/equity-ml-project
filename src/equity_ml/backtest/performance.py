"""Performance evaluation (MODEL_DESIGN.md Section 8): return-based metrics
(gross and net of cost), a transaction-cost sensitivity table, and a
factor-exposure regression against the Fama-French 5 + momentum factors
already merged into the master panel — reporting alpha (the return
unexplained by known factors) and factor loadings, so a result that's "just
momentum in disguise" is visible rather than mistaken for new signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from equity_ml.backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: int) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    if len(downside) < 2 or downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(returns.mean() / downside_std * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def compute_performance_metrics(
    portfolio_returns: pd.DataFrame, config: BacktestConfig | None = None
) -> pd.DataFrame:
    """One row per model: annualized return/vol, Sharpe, Sortino, max
    drawdown, and hit rate — computed separately for gross and net-of-cost
    return series (MODEL_DESIGN.md Section 8)."""
    config = config or BacktestConfig()
    rows = []
    for model, group in portfolio_returns.groupby(config.model_column):
        group = group.sort_values(config.date_column)
        for kind, column in (("gross", "gross_return"), ("net", "net_return")):
            returns = group[column]
            n = len(returns)
            rows.append(
                {
                    "model": model,
                    "kind": kind,
                    "annualized_return": float(returns.mean() * config.periods_per_year),
                    "annualized_vol": float(returns.std(ddof=1) * np.sqrt(config.periods_per_year)),
                    "sharpe": _sharpe(returns, config.periods_per_year),
                    "sortino": _sortino(returns, config.periods_per_year),
                    "max_drawdown": _max_drawdown(returns),
                    "hit_rate": float((returns > 0).mean()) if n else float("nan"),
                    "n_periods": n,
                }
            )
    return pd.DataFrame(rows)


def cost_sensitivity_table(
    portfolio_returns: pd.DataFrame, config: BacktestConfig | None = None
) -> pd.DataFrame:
    """Net Sharpe ratio at each of `config.cost_sensitivity_bps`, per model
    — shows how cost-sensitive each model's edge is, rather than trusting
    one hardcoded transaction-cost assumption (MODEL_DESIGN.md Section 7)."""
    config = config or BacktestConfig()
    rows = []
    for model, group in portfolio_returns.groupby(config.model_column):
        group = group.sort_values(config.date_column)
        for bps in config.cost_sensitivity_bps:
            cost = group["turnover"].fillna(0.0) * (bps / 10_000.0)
            net = group["gross_return"] - cost
            rows.append(
                {
                    "model": model,
                    "cost_bps": bps,
                    "sharpe": _sharpe(net, config.periods_per_year),
                    "annualized_return": float(net.mean() * config.periods_per_year),
                }
            )
    return pd.DataFrame(rows)


def factor_regression(
    portfolio_returns: pd.DataFrame,
    factor_panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    return_column: str = "net_return",
) -> pd.DataFrame:
    """Regress each model's long-short return series on the Fama-French
    factors (`config.factor_columns`, already point-in-time merged onto the
    master panel by data_processing). Reports alpha (intercept), factor
    loadings, and Newey-West (HAC) t-statistics — robust to the
    autocorrelation a monthly long-short return series can have.

    `factor_panel` must have `config.date_column` plus every column in
    `config.factor_columns`, one row per month (e.g. the master panel's
    `ff_*` columns, deduplicated to one row per date).
    """
    config = config or BacktestConfig()
    factors = factor_panel[[config.date_column, *config.factor_columns]].drop_duplicates(
        subset=config.date_column
    )
    rows = []
    for model, group in portfolio_returns.groupby(config.model_column):
        merged = group.merge(factors, on=config.date_column, how="inner").sort_values(
            config.date_column
        )
        if len(merged) < len(config.factor_columns) + 5:
            logger.warning("%s: too few overlapping months for factor regression, skipping", model)
            continue
        y = merged[return_column].to_numpy(dtype="float64")
        x = sm.add_constant(merged[list(config.factor_columns)].to_numpy(dtype="float64"))
        max_lags = max(1, round(4 * (len(merged) / 100.0) ** (2.0 / 9.0)))
        fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

        row: dict[str, object] = {
            "model": model,
            "n_months": len(merged),
            "alpha": fit.params[0],
            "alpha_t_stat": fit.tvalues[0],
        }
        for i, factor_name in enumerate(config.factor_columns, start=1):
            row[f"{factor_name}_beta"] = fit.params[i]
            row[f"{factor_name}_t_stat"] = fit.tvalues[i]
        row["r_squared"] = fit.rsquared
        rows.append(row)
    return pd.DataFrame(rows)
