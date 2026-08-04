"""Orchestrates the full Phase 4 backtest build (MODEL_DESIGN.md Sections
6-8): load out-of-sample predictions, build long/short portfolio weights,
apply realized returns, compute performance metrics + cost sensitivity +
factor-exposure regression, and write reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from equity_ml.backtest.config import BacktestConfig
from equity_ml.backtest.engine import compute_portfolio_returns
from equity_ml.backtest.io import load_predictions, load_realized_returns_and_factors
from equity_ml.backtest.performance import (
    compute_performance_metrics,
    cost_sensitivity_table,
    factor_regression,
)
from equity_ml.backtest.portfolio import build_portfolio_weights
from equity_ml.backtest.reporting import write_reports
from equity_ml.models.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestPipelineResult:
    summary_path: Path


def run_pipeline(
    processed_dir: Path | str,
    modeling_reports_dir: Path | str,
    reports_dir: Path | str,
    model_config: ModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> BacktestPipelineResult:
    model_config = model_config or ModelConfig()
    backtest_config = backtest_config or BacktestConfig()

    logger.info("Loading out-of-sample predictions")
    predictions = load_predictions(modeling_reports_dir)

    logger.info("Loading realized returns and factor data")
    realized_returns, factor_panel = load_realized_returns_and_factors(
        processed_dir, model_config, backtest_config
    )

    logger.info("Building long/short portfolio weights")
    weights = build_portfolio_weights(predictions, backtest_config)

    logger.info("Computing portfolio returns")
    portfolio_returns = compute_portfolio_returns(weights, realized_returns, backtest_config)

    logger.info("Computing performance metrics")
    performance_metrics = compute_performance_metrics(portfolio_returns, backtest_config)
    cost_sensitivity = cost_sensitivity_table(portfolio_returns, backtest_config)
    factor_regression_results = factor_regression(portfolio_returns, factor_panel, backtest_config)

    summary_path = write_reports(
        portfolio_returns,
        performance_metrics,
        cost_sensitivity,
        factor_regression_results,
        reports_dir,
        backtest_config,
    )
    return BacktestPipelineResult(summary_path=summary_path)
