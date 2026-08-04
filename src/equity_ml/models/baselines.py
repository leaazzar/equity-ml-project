"""Benchmark models (MODEL_DESIGN.md Section 4), weakest to strongest, so an
ML model's value-add is measurable against something rather than against "no
model." Every baseline shares the same runner signature as the ML model
runners in `training.py` — `(train_df, inner_train_df, inner_val_df,
test_df, feature_columns, target_column, config) -> (predictions, params)`
— so `training.run_walk_forward_training` can score them through the
identical walk-forward harness as every other model, per MODEL_DESIGN.md's
"no benchmark gets a different, more favorable evaluation protocol."
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from equity_ml.models.config import ModelConfig

logger = logging.getLogger(__name__)

RunnerResult = tuple[pd.Series, dict[str, Any]]


def equal_weight_runner(
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: ModelConfig,
) -> RunnerResult:
    """Zero-skill benchmark: every stock gets an identical score. A
    dollar-neutral long-short portfolio built from this should realize ~0
    return in expectation — establishes the floor every other model is
    measured against."""
    del train_df, inner_train_df, inner_val_df, feature_columns, target_column, config
    return pd.Series(0.0, index=test_df.index), {}


def momentum_sort_runner(
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: ModelConfig,
) -> RunnerResult:
    """Single-factor benchmark: rank by `config.momentum_benchmark_feature`
    (12-1 momentum by default) alone. A model that can't beat this on a
    risk-adjusted basis isn't adding value over a one-line factor sort."""
    del train_df, inner_train_df, inner_val_df, feature_columns, target_column
    col = config.momentum_benchmark_feature
    rows = test_df.dropna(subset=[col])
    return rows[col], {"feature": col}


def fama_macbeth_runner(
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: ModelConfig,
) -> RunnerResult:
    """Fama-MacBeth cross-sectional regression: fit one OLS per month in the
    fold's full outer training window (`train_df` — no inner validation
    split needed, since there are no hyperparameters to tune), average the
    coefficients over time, and apply that fixed linear model to the test
    window. The standard academic linear benchmark."""
    del inner_train_df, inner_val_df
    monthly_betas = []
    for _month, group in train_df.groupby(config.date_column):
        complete = group.dropna(subset=[*feature_columns, target_column])
        if len(complete) < len(feature_columns) + 5:
            continue
        design = np.column_stack(
            [np.ones(len(complete)), complete[feature_columns].to_numpy(dtype="float64")]
        )
        outcome = complete[target_column].to_numpy(dtype="float64")
        beta, *_ = np.linalg.lstsq(design, outcome, rcond=None)
        monthly_betas.append(beta)

    rows = test_df.dropna(subset=feature_columns)
    if not monthly_betas:
        logger.warning("fama_macbeth: no training month had enough complete-case rows to fit")
        return pd.Series(np.nan, index=rows.index), {"n_months_fit": 0}

    avg_beta = np.mean(monthly_betas, axis=0)
    design = np.column_stack([np.ones(len(rows)), rows[feature_columns].to_numpy(dtype="float64")])
    preds = design @ avg_beta
    return pd.Series(preds, index=rows.index), {"n_months_fit": len(monthly_betas)}


DEFAULT_BASELINE_RUNNERS: dict[str, Any] = {
    "equal_weight": equal_weight_runner,
    "momentum_sort": momentum_sort_runner,
    "fama_macbeth": fama_macbeth_runner,
}
