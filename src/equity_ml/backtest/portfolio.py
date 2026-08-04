"""Predicted scores -> long/short portfolio weights (MODEL_DESIGN.md Section
6): rank each month's cross-section by predicted score, long the top decile,
short the bottom decile, equal- or score-weighted within each leg,
dollar-neutral (100% long / 100% short, 0% net) by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from equity_ml.backtest.config import BacktestConfig

logger = logging.getLogger(__name__)


def _weights_for_one_month(scores: pd.Series, config: BacktestConfig) -> pd.Series:
    """Long/short weights for one month's cross-section of scores. Weights
    sum to +1.0 across the long leg and -1.0 across the short leg (0 net
    exposure); every other row gets 0."""
    if config.decile_count < 2:
        raise ValueError(
            f"decile_count must be >= 2 to form distinct long/short legs, got {config.decile_count}"
        )
    weights = pd.Series(0.0, index=scores.index)
    valid = scores.dropna()
    if len(valid) < 2 * config.decile_count:
        # Too few names to form config.decile_count non-trivial buckets.
        return weights

    ranks = valid.rank(method="first")
    bucket = (
        np.ceil(ranks / len(valid) * config.decile_count).astype(int).clip(1, config.decile_count)
    )
    long_mask = bucket == config.decile_count
    short_mask = bucket == 1

    if config.weighting_scheme == "equal":
        if long_mask.any():
            weights.loc[valid.index[long_mask]] = 1.0 / long_mask.sum()
        if short_mask.any():
            weights.loc[valid.index[short_mask]] = -1.0 / short_mask.sum()
    elif config.weighting_scheme == "score":
        long_scores = valid[long_mask].clip(lower=0)
        if long_scores.sum() > 0:
            weights.loc[long_scores.index] = long_scores / long_scores.sum()
        short_scores = (-valid[short_mask]).clip(lower=0)
        if short_scores.sum() > 0:
            weights.loc[short_scores.index] = -(short_scores / short_scores.sum())
    else:
        raise ValueError(f"Unknown weighting_scheme: {config.weighting_scheme!r}")

    return weights


def build_portfolio_weights(
    predictions: pd.DataFrame, config: BacktestConfig | None = None
) -> pd.DataFrame:
    """Build long/short weights for every (model, date) cross-section in
    `predictions` (the out-of-sample prediction panel from
    `equity_ml.models.training.run_walk_forward_training`).

    Returns `[permno_column, date_column, model_column, weight]`, one row
    per (permno, date, model) with a non-zero weight kept only for rows the
    portfolio actually holds (zero-weight rows are dropped, not kept as
    explicit zeros, to keep the output proportional to actual holdings).
    """
    config = config or BacktestConfig()
    records = []
    for (date, model), group in predictions.groupby([config.date_column, config.model_column]):
        weights = _weights_for_one_month(
            group.set_index(config.permno_column)[config.score_column], config
        )
        nonzero = weights[weights != 0.0]
        if nonzero.empty:
            continue
        records.append(
            pd.DataFrame(
                {
                    config.permno_column: nonzero.index,
                    config.date_column: date,
                    config.model_column: model,
                    "weight": nonzero.to_numpy(),
                }
            )
        )
    if not records:
        return pd.DataFrame(
            columns=[config.permno_column, config.date_column, config.model_column, "weight"]
        )
    return pd.concat(records, ignore_index=True)
