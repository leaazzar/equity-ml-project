"""Explainability and signal-quality diagnostics (MODEL_DESIGN.md Sections
8-9): per-fold/pooled Information Coefficient series, permutation feature
importance, and category-level attribution using the same nine feature
categories already defined in `feature_engineering.registry`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from equity_ml.models.config import ModelConfig
from equity_ml.models.estimators import EstimatorSpec
from equity_ml.models.splits import WalkForwardFold, slice_by_dates
from equity_ml.models.training import fit_final_model, training_rows
from equity_ml.models.tuning import information_coefficient
from feature_engineering.registry import feature_names_by_category

logger = logging.getLogger(__name__)


def compute_ic_series(predictions: pd.DataFrame, config: ModelConfig | None = None) -> pd.DataFrame:
    """Monthly cross-sectional IC per model (MODEL_DESIGN.md Section 8).
    Returns columns [date_column, model, ic, n]."""
    config = config or ModelConfig()
    records = []
    for (date, model), group in predictions.groupby([config.date_column, "model"]):
        ic = information_coefficient(group["y_true"], group["y_pred"].to_numpy())
        records.append({config.date_column: date, "model": model, "ic": ic, "n": len(group)})
    return (
        pd.DataFrame.from_records(records)
        .sort_values([config.date_column, "model"])
        .reset_index(drop=True)
    )


def summarize_ic(ic_series: pd.DataFrame, config: ModelConfig | None = None) -> pd.DataFrame:
    """Per-model pooled IC summary: mean, std, t-stat (mean / (std / sqrt(n))),
    and number of months — the headline signal-quality comparison table
    (MODEL_DESIGN.md Section 8)."""
    config = config or ModelConfig()
    rows = []
    for model, group in ic_series.groupby("model"):
        values = group["ic"].dropna()
        n = len(values)
        mean = values.mean() if n else float("nan")
        std = values.std(ddof=1) if n > 1 else float("nan")
        t_stat = (
            mean / (std / np.sqrt(n))
            if n > 1 and std not in (0, None) and not np.isnan(std)
            else float("nan")
        )
        rows.append(
            {"model": model, "ic_mean": mean, "ic_std": std, "ic_t_stat": t_stat, "n_months": n}
        )
    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False).reset_index(drop=True)


def permutation_importance_by_fold(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    folds: list[WalkForwardFold],
    spec: EstimatorSpec,
    config: ModelConfig | None = None,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Refit `spec` per fold (same tune-then-fit procedure `training.py`
    uses) and compute permutation importance on that fold's inner
    validation split — never the fold's own test window, so importance
    estimates carry the same no-test-leakage guarantee as the predictions
    themselves. Returns long-format [fold_id, feature, importance_mean,
    importance_std]."""
    config = config or ModelConfig()
    records = []
    for fold in folds:
        train_df = slice_by_dates(panel, fold.train_start, fold.train_end, config)
        inner_train_df = slice_by_dates(panel, fold.train_start, fold.inner_train_end, config)
        inner_val_df = slice_by_dates(panel, fold.inner_val_start, fold.inner_val_end, config)

        model, _params, _ic = fit_final_model(
            spec, train_df, inner_train_df, inner_val_df, feature_columns, target_column, config
        )
        val = training_rows(inner_val_df, feature_columns, target_column)
        if len(val) == 0:
            logger.warning(
                "fold %d: empty inner validation split, skipping importance", fold.fold_id
            )
            continue
        result = permutation_importance(
            model,
            val[feature_columns],
            val[target_column],
            n_repeats=n_repeats,
            random_state=config.random_seed,
            scoring="neg_mean_squared_error",
        )
        for i, feature in enumerate(feature_columns):
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "feature": feature,
                    "importance_mean": result.importances_mean[i],
                    "importance_std": result.importances_std[i],
                }
            )
    return pd.DataFrame(records)


def category_importance(importances: pd.DataFrame) -> pd.DataFrame:
    """Roll individual-feature permutation importances up to
    `feature_engineering.registry`'s nine categories (size, value, momentum,
    reversal, quality, growth, leverage, liquidity, volatility) — a more
    economically legible summary than a 50-feature list (MODEL_DESIGN.md
    Section 9)."""
    if importances.empty:
        return pd.DataFrame(columns=["category", "importance_mean"])
    by_category = feature_names_by_category()
    feature_to_category = {
        feature: category for category, features in by_category.items() for feature in features
    }
    with_category = importances.copy()
    with_category["category"] = with_category["feature"].map(feature_to_category)
    return (
        with_category.groupby("category")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
