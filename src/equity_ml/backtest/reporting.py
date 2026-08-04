"""Writes Phase 4 backtest diagnostics under reports/backtest/ (gitignored,
derived from the licensed WRDS extract — same convention as
`feature_engineering.reporting` / `equity_ml.models.reporting`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from equity_ml.backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


def write_reports(
    portfolio_returns: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    factor_regression_results: pd.DataFrame,
    reports_dir: Path | str,
    config: BacktestConfig | None = None,
) -> Path:
    """Write all backtest diagnostics and summary.md. Returns summary.md's path."""
    config = config or BacktestConfig()
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    portfolio_returns.to_csv(reports_dir / "portfolio_returns.csv", index=False)
    performance_metrics.to_csv(reports_dir / "performance_metrics.csv", index=False)
    cost_sensitivity.to_csv(reports_dir / "cost_sensitivity.csv", index=False)
    factor_regression_results.to_csv(reports_dir / "factor_regression.csv", index=False)

    generated_at = datetime.now(UTC).isoformat()

    net_metrics = performance_metrics[performance_metrics["kind"] == "net"].sort_values(
        "sharpe", ascending=False
    )
    perf_table = "\n".join(
        f"| {row.model} | {row.annualized_return:.4f} | {row.annualized_vol:.4f} | "
        f"{row.sharpe:.2f} | {row.sortino:.2f} | {row.max_drawdown:.4f} | {row.hit_rate:.2f} |"
        for row in net_metrics.itertuples()
    )

    factor_table = "\n".join(
        f"| {row.model} | {row.alpha:.4f} | {row.alpha_t_stat:.2f} | {row.r_squared:.3f} |"
        for row in factor_regression_results.itertuples()
    )

    body = f"""# Backtest Diagnostics

Generated: {generated_at}

Transaction cost assumption: {config.transaction_cost_bps:.1f} bps one-way
(see cost_sensitivity.csv for the full 0/5/10/20/50 bps sensitivity table —
this project has no bid-ask-spread or commission data, so this is a
documented placeholder, not a measured figure).

## Net-of-cost performance (ranked by Sharpe)

| Model | Ann. Return | Ann. Vol | Sharpe | Sortino | Max Drawdown | Hit Rate |
| --- | --- | --- | --- | --- | --- | --- |
{perf_table}

## Factor-exposure regression (alpha, net-of-cost returns)

| Model | Alpha (monthly) | Alpha t-stat | R^2 |
| --- | --- | --- | --- |
{factor_table}

## Files in this directory

- `portfolio_returns.csv` — gross/net return, turnover, cost per (model, date).
- `performance_metrics.csv` — this file's performance table, gross and net.
- `cost_sensitivity.csv` — Sharpe/annualized return at each cost-bps level.
- `factor_regression.csv` — this file's factor-regression table, full detail.
"""
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(body)
    logger.info("Wrote backtest diagnostics to %s", reports_dir)
    return summary_path
