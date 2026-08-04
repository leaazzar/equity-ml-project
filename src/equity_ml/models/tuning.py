"""Hyperparameter search (MODEL_DESIGN.md Section 5) — a small explicit grid
search selected by validation-fold Information Coefficient (Spearman rank
correlation between predicted score and realized forward return), not MSE.
IC directly measures the cross-sectional ranking quality a long-short
portfolio (MODEL_DESIGN.md Section 6) actually monetizes; MSE rewards
getting a noisy return's magnitude right, which is a different objective.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from equity_ml.models.config import ModelConfig
from equity_ml.models.estimators import EstimatorSpec

logger = logging.getLogger(__name__)


def information_coefficient(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and realized outcomes.
    Rows with a missing `y_true` (e.g. a benchmark scored on more rows than
    it has valid forward-return labels for) are excluded first — scipy's
    `spearmanr` otherwise returns NaN for the *entire* month if even one
    pair is missing, silently discarding an otherwise-valid IC estimate.
    NaN if fewer than 2 valid pairs remain (undefined correlation)."""
    y_true_array = np.asarray(y_true, dtype="float64")
    y_pred_array = np.asarray(y_pred, dtype="float64")
    valid = ~(np.isnan(y_true_array) | np.isnan(y_pred_array))
    if valid.sum() < 2:
        return float("nan")
    corr, _ = spearmanr(y_true_array[valid], y_pred_array[valid])
    return float(corr)


def _param_grid_combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*param_grid.values())]


def tune_estimator(
    spec: EstimatorSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    config: ModelConfig | None = None,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Grid search `spec.param_grid`, fitting on (x_train, y_train) and
    scoring by IC on (x_val, y_val). Returns (best_params, best_ic,
    all_results) — `all_results` is every candidate's params + IC, for
    diagnostics (MODEL_DESIGN.md Section 9 doesn't require this, but it's
    useful for debugging a tuning run and costs nothing extra to keep).

    If every candidate produces a NaN or non-improving IC (e.g. a degenerate
    validation split), falls back to the grid's first candidate rather than
    leaving `best_params` undefined.
    """
    config = config or ModelConfig()
    best_params: dict[str, Any] | None = None
    best_ic = -np.inf
    results: list[dict[str, Any]] = []

    for params in _param_grid_combinations(spec.param_grid):
        model = spec.build(params, config.random_seed)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        ic = information_coefficient(y_val, preds)
        results.append({"params": params, "ic": ic})
        if not np.isnan(ic) and ic > best_ic:
            best_ic = ic
            best_params = params

    if best_params is None:
        best_params = _param_grid_combinations(spec.param_grid)[0]
        best_ic = float("nan")
        logger.warning(
            "%s: every hyperparameter candidate scored NaN/non-improving IC; "
            "falling back to the first grid candidate %s",
            spec.name,
            best_params,
        )

    return best_params, best_ic, results
