"""Walk-forward training orchestration (MODEL_DESIGN.md Sections 3-5):
for every fold and every model (benchmark or ML), tune on the fold's inner
train/validation split, fit on the fold's full outer training window, and
predict on the fold's test window. Concatenating every fold's test
predictions yields one out-of-sample prediction panel spanning the full
walk-forward evaluation period — what MODEL_DESIGN.md Sections 6-8 consume.

No hyperparameter or model-family decision is ever made by looking at a
test fold (MODEL_DESIGN.md Section 2) — `run_walk_forward_training` never
passes `test_df` into `tune_estimator`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin

from equity_ml.models.config import ModelConfig
from equity_ml.models.estimators import EstimatorSpec
from equity_ml.models.splits import WalkForwardFold, slice_by_dates
from equity_ml.models.tuning import tune_estimator

logger = logging.getLogger(__name__)

ModelRunner = Callable[
    [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str, ModelConfig],
    tuple[pd.Series, dict[str, Any]],
]


def training_rows(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    """Complete-case rows only — no feature is ever imputed anywhere in this
    codebase (FEATURE_DICTIONARY.md), and that convention extends to model
    training here. Applied uniformly across every model family (including
    HistGradientBoostingRegressor, which natively tolerates missing values)
    so every model in the comparison table is fit on the exact same rows."""
    return df.dropna(subset=[*feature_columns, target_column])


def fit_final_model(
    spec: EstimatorSpec,
    train_df: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    inner_val_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: ModelConfig,
) -> tuple[RegressorMixin, dict[str, Any], float]:
    """Tune `spec` on the inner train/validation split, then fit the chosen
    hyperparameters on the full outer training window. Returns (fitted
    model, best hyperparameters, inner-validation IC). Shared by
    `make_ml_model_runner` (prediction) and `diagnostics.py` (permutation
    importance needs the fitted model itself, not just its predictions)."""
    inner_train = training_rows(inner_train_df, feature_columns, target_column)
    inner_val = training_rows(inner_val_df, feature_columns, target_column)

    if len(inner_train) == 0 or len(inner_val) == 0:
        best_params: dict[str, Any] = {}
        best_ic = float("nan")
        logger.warning("%s: empty inner train/val split, using default hyperparameters", spec.name)
    else:
        best_params, best_ic, _ = tune_estimator(
            spec,
            inner_train[feature_columns],
            inner_train[target_column],
            inner_val[feature_columns],
            inner_val[target_column],
            config,
        )

    full_train = training_rows(train_df, feature_columns, target_column)
    model = spec.build(best_params, config.random_seed)
    model.fit(full_train[feature_columns], full_train[target_column])
    return model, best_params, best_ic


def make_ml_model_runner(spec: EstimatorSpec) -> ModelRunner:
    """Wrap an `EstimatorSpec` into the common model-runner signature: tune
    on the fold's inner split, refit on the full outer training window, and
    predict every test-window row with complete features (target need not be
    known to predict — only to train or to later score)."""

    def run(
        train_df: pd.DataFrame,
        inner_train_df: pd.DataFrame,
        inner_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        config: ModelConfig,
    ) -> tuple[pd.Series, dict[str, Any]]:
        model, best_params, best_ic = fit_final_model(
            spec, train_df, inner_train_df, inner_val_df, feature_columns, target_column, config
        )
        predict_rows = test_df.dropna(subset=feature_columns)
        preds = pd.Series(model.predict(predict_rows[feature_columns]), index=predict_rows.index)
        return preds, {"params": best_params, "inner_val_ic": best_ic}

    return run


def run_walk_forward_training(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    folds: list[WalkForwardFold],
    model_runners: dict[str, ModelRunner],
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Run every model runner through every walk-forward fold. Returns a
    single concatenated out-of-sample prediction panel with columns:
    `[permno_column, date_column, model, fold_id, y_true, y_pred]`.
    """
    config = config or ModelConfig()
    records: list[pd.DataFrame] = []

    for fold in folds:
        train_df = slice_by_dates(panel, fold.train_start, fold.train_end, config)
        inner_train_df = slice_by_dates(panel, fold.train_start, fold.inner_train_end, config)
        inner_val_df = slice_by_dates(panel, fold.inner_val_start, fold.inner_val_end, config)
        test_df = slice_by_dates(panel, fold.test_start, fold.test_end, config)

        for name, runner in model_runners.items():
            preds, info = runner(
                train_df,
                inner_train_df,
                inner_val_df,
                test_df,
                feature_columns,
                target_column,
                config,
            )
            fold_records = test_df.loc[
                preds.index, [config.permno_column, config.date_column, target_column]
            ].copy()
            fold_records = fold_records.rename(columns={target_column: "y_true"})
            fold_records["model"] = name
            fold_records["fold_id"] = fold.fold_id
            fold_records["y_pred"] = preds
            records.append(fold_records)
            logger.info(
                "Fold %d [%s .. %s], model=%s: %d predictions (%s)",
                fold.fold_id,
                fold.test_start.date(),
                fold.test_end.date(),
                name,
                len(preds),
                info,
            )

    if not records:
        return pd.DataFrame(
            columns=[
                config.permno_column,
                config.date_column,
                "model",
                "fold_id",
                "y_true",
                "y_pred",
            ]
        )
    return pd.concat(records, ignore_index=True)
