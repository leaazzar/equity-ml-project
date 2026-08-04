from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_ml.models.config import ModelConfig
from equity_ml.models.estimators import DEFAULT_ESTIMATORS, RIDGE
from equity_ml.models.tuning import information_coefficient, tune_estimator


def test_information_coefficient_perfect_rank_correlation() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)


def test_information_coefficient_perfect_inverse_correlation() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    assert information_coefficient(y_true, y_pred) == pytest.approx(-1.0)


def test_information_coefficient_too_few_observations_is_nan() -> None:
    assert np.isnan(information_coefficient(pd.Series([1.0]), np.array([1.0])))


def test_information_coefficient_ignores_nan_in_y_true_rather_than_propagating() -> None:
    # A single missing y_true must not silently NaN out an otherwise valid
    # month's IC estimate (real bug found running against real data: a
    # benchmark scored on more rows than it had valid labels for produced
    # NaN IC for every single month).
    y_true = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 999.0, 40.0, 50.0])
    assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)


def test_information_coefficient_ignores_nan_in_y_pred_too() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, np.nan, 40.0, 50.0])
    assert information_coefficient(y_true, y_pred) == pytest.approx(1.0)


def test_tune_estimator_selects_grid_member_with_best_validation_ic() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 1))
    y = 2.0 * x[:, 0] + rng.normal(scale=0.01, size=n)
    x_train = pd.DataFrame({"f1": x[:, 0]})
    y_train = pd.Series(y)
    x_val = x_train
    y_val = y_train

    best_params, best_ic, results = tune_estimator(
        RIDGE, x_train, y_train, x_val, y_val, ModelConfig()
    )
    assert best_params["alpha"] in RIDGE.param_grid["alpha"]
    assert best_ic == pytest.approx(1.0, abs=0.05)
    assert len(results) == len(RIDGE.param_grid["alpha"])


def test_tune_estimator_falls_back_when_every_candidate_is_degenerate() -> None:
    # A single-row validation split makes IC undefined (NaN) for every candidate.
    x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    x_val = pd.DataFrame({"f1": [1.0]})
    y_val = pd.Series([1.0])

    best_params, best_ic, _ = tune_estimator(RIDGE, x_train, y_train, x_val, y_val, ModelConfig())
    assert best_params == {"alpha": RIDGE.param_grid["alpha"][0]}
    assert np.isnan(best_ic)


@pytest.mark.parametrize("spec", DEFAULT_ESTIMATORS, ids=lambda s: s.name)
def test_every_default_estimator_builds_fits_and_predicts(spec) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    y = pd.Series(rng.normal(size=40))
    first_params = {key: values[0] for key, values in spec.param_grid.items()}
    model = spec.build(first_params, seed=0)
    model.fit(x, y)
    preds = model.predict(x)
    assert len(preds) == len(x)
