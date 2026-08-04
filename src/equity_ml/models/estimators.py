"""ML model factories and hyperparameter search grids (MODEL_DESIGN.md
Sections 4-5). Every estimator here is available from scikit-learn, already
a project dependency (`pyproject.toml`) — no new dependency (e.g.
LightGBM/XGBoost) is added; see MODEL_DESIGN.md Section 4 for why that's a
deliberate fast-follow, not part of this first pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge


@dataclass(frozen=True)
class EstimatorSpec:
    """One model family: a name, a factory (`hyperparams, seed -> estimator`),
    and the small hyperparameter grid MODEL_DESIGN.md Section 5 tunes over.
    """

    name: str
    build: Callable[[dict[str, Any], int], RegressorMixin]
    param_grid: dict[str, list[Any]]


def _build_ridge(params: dict[str, Any], seed: int) -> Ridge:
    return Ridge(random_state=seed, **params)


def _build_lasso(params: dict[str, Any], seed: int) -> Lasso:
    return Lasso(random_state=seed, **params)


def _build_elastic_net(params: dict[str, Any], seed: int) -> ElasticNet:
    return ElasticNet(random_state=seed, **params)


def _build_random_forest(params: dict[str, Any], seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=seed, n_estimators=300, n_jobs=-1, **params)


def _build_hist_gradient_boosting(
    params: dict[str, Any], seed: int
) -> HistGradientBoostingRegressor:
    # Natively tolerates missing feature values, unlike the other estimators
    # here — the walk-forward pipeline still applies uniform complete-case
    # filtering across all models (training.py) so every model family is
    # compared on the exact same rows.
    return HistGradientBoostingRegressor(random_state=seed, **params)


RIDGE = EstimatorSpec(
    name="ridge",
    build=_build_ridge,
    param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
)

LASSO = EstimatorSpec(
    name="lasso",
    build=_build_lasso,
    param_grid={"alpha": [0.0001, 0.001, 0.01, 0.1]},
)

ELASTIC_NET = EstimatorSpec(
    name="elastic_net",
    build=_build_elastic_net,
    param_grid={"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
)

RANDOM_FOREST = EstimatorSpec(
    name="random_forest",
    build=_build_random_forest,
    param_grid={
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [50, 200, 1000],
        "max_features": ["sqrt", 0.5],
    },
)

HIST_GRADIENT_BOOSTING = EstimatorSpec(
    name="hist_gradient_boosting",
    build=_build_hist_gradient_boosting,
    param_grid={
        "max_depth": [3, 5, None],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_regularization": [0.0, 1.0],
    },
)

DEFAULT_ESTIMATORS: tuple[EstimatorSpec, ...] = (
    RIDGE,
    LASSO,
    ELASTIC_NET,
    RANDOM_FOREST,
    HIST_GRADIENT_BOOSTING,
)
