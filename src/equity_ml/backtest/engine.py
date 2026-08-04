"""Applies portfolio weights to realized returns (MODEL_DESIGN.md Section
7): gross portfolio return, turnover, and net-of-cost return per (model,
date). Uses `ret_adj` (via `realized_return_column`, populated from the
same delisting-adjusted return the target was built from) — not a naive
`MthRet`-only return — so the backtest doesn't silently benefit from
ignoring delisting losses (MERGE_REPORT.md cites Shumway 1997 on exactly
this bias).
"""

from __future__ import annotations

import logging

import pandas as pd

from equity_ml.backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


def _turnover_per_model(weights: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Turnover at each rebalance date for one model: half the sum of
    absolute weight changes versus the prior rebalance date (missing
    positions on either side treated as 0 weight). The first rebalance
    date's turnover is the cost of building the portfolio from cash (half
    the sum of absolute initial weights) — a real cost, not excluded."""
    by_date: dict[pd.Timestamp, pd.Series] = {
        pd.Timestamp(date): group.set_index(config.permno_column)["weight"]  # type: ignore[arg-type]
        for date, group in weights.groupby(config.date_column)
    }
    dates = sorted(by_date)
    turnover = {}
    prev_weights: pd.Series | None = None
    for date in dates:
        current = by_date[date]
        if prev_weights is None:
            turnover[date] = 0.5 * current.abs().sum()
        else:
            combined_index = current.index.union(prev_weights.index)
            diff = current.reindex(combined_index, fill_value=0.0) - prev_weights.reindex(
                combined_index, fill_value=0.0
            )
            turnover[date] = 0.5 * diff.abs().sum()
        prev_weights = current
    return pd.Series(turnover, name="turnover")


def compute_portfolio_returns(
    weights: pd.DataFrame, realized_returns: pd.DataFrame, config: BacktestConfig | None = None
) -> pd.DataFrame:
    """Combine portfolio weights with realized returns into per-(model,
    date) gross/net returns and turnover.

    `realized_returns` must have `[permno_column, date_column,
    realized_return_column]` — the target's raw (non-demeaned) forward
    return, so realized portfolio return reflects actual dollar P&L, not
    the market-neutralized quantity the model was trained on.

    Returns `[date_column, model_column, gross_return, turnover, cost,
    net_return]`, one row per (model, date) with at least one holding.
    """
    config = config or BacktestConfig()
    merged = weights.merge(
        realized_returns[[config.permno_column, config.date_column, config.realized_return_column]],
        on=[config.permno_column, config.date_column],
        how="left",
    )
    n_missing_return = int(merged[config.realized_return_column].isna().sum())
    if n_missing_return:
        logger.warning(
            "%d / %d weighted positions have no realized return (dropped from portfolio "
            "return calculation)",
            n_missing_return,
            len(merged),
        )
    merged = merged.dropna(subset=[config.realized_return_column])
    merged["contribution"] = merged["weight"] * merged[config.realized_return_column]

    gross = merged.groupby([config.date_column, config.model_column])["contribution"].sum()
    gross.name = "gross_return"

    records = []
    for model, model_weights in weights.groupby(config.model_column):
        turnover = _turnover_per_model(model_weights, config)
        turnover_df = turnover.reset_index()
        turnover_df.columns = [config.date_column, "turnover"]
        turnover_df[config.model_column] = model
        records.append(turnover_df)
    turnover_all = (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame(columns=[config.date_column, "turnover", config.model_column])
    )

    result = gross.reset_index().merge(
        turnover_all, on=[config.date_column, config.model_column], how="left"
    )
    result["cost"] = result["turnover"].fillna(0.0) * (config.transaction_cost_bps / 10_000.0)
    result["net_return"] = result["gross_return"] - result["cost"]
    return result.sort_values([config.model_column, config.date_column]).reset_index(drop=True)
