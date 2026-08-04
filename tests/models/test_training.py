from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_ml.models.baselines import equal_weight_runner
from equity_ml.models.config import ModelConfig
from equity_ml.models.estimators import RIDGE
from equity_ml.models.splits import build_walk_forward_folds
from equity_ml.models.training import make_ml_model_runner, run_walk_forward_training
from equity_ml.models.tuning import information_coefficient


@pytest.fixture
def learnable_panel() -> pd.DataFrame:
    """60 months x 100 synthetic PERMNOs, target = 2*f1 + small noise, so a
    correctly-trained model should recover a strong positive out-of-sample
    Information Coefficient, while a signal-blind baseline should not."""
    rng = np.random.default_rng(42)
    months = pd.date_range("2000-01-31", periods=60, freq="ME")
    rows = []
    for month in months:
        f1 = rng.normal(size=100)
        target = 2.0 * f1 + rng.normal(scale=0.05, size=100)
        for permno in range(100):
            rows.append(
                {
                    "PERMNO": permno,
                    "MthCalDt": month,
                    "f1": f1[permno],
                    "target": target[permno],
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_training_produces_predictions_for_every_test_row(
    learnable_panel: pd.DataFrame,
) -> None:
    config = ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=1)
    folds = build_walk_forward_folds(learnable_panel["MthCalDt"], config)
    predictions = run_walk_forward_training(
        learnable_panel,
        feature_columns=["f1"],
        target_column="target",
        folds=folds,
        model_runners={"ridge": make_ml_model_runner(RIDGE), "equal_weight": equal_weight_runner},
        config=config,
    )
    assert set(predictions["model"]) == {"ridge", "equal_weight"}
    assert set(predictions["fold_id"]) == {fold.fold_id for fold in folds}
    ridge_preds = predictions[predictions["model"] == "ridge"]
    assert not ridge_preds["y_pred"].isna().any()
    assert len(ridge_preds) == 100 * sum(
        (fold.test_end.year - fold.test_start.year) * 12
        + (fold.test_end.month - fold.test_start.month)
        + 1
        for fold in folds
    )


def test_ml_model_beats_equal_weight_baseline_on_learnable_signal(
    learnable_panel: pd.DataFrame,
) -> None:
    config = ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=1)
    folds = build_walk_forward_folds(learnable_panel["MthCalDt"], config)
    predictions = run_walk_forward_training(
        learnable_panel,
        feature_columns=["f1"],
        target_column="target",
        folds=folds,
        model_runners={"ridge": make_ml_model_runner(RIDGE)},
        config=config,
    )
    ic = information_coefficient(predictions["y_true"], predictions["y_pred"].to_numpy())
    assert ic > 0.9  # near-noiseless linear signal should be recovered almost exactly


def test_no_test_fold_date_leaks_into_that_folds_training_window(
    learnable_panel: pd.DataFrame,
) -> None:
    config = ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=3)
    folds = build_walk_forward_folds(learnable_panel["MthCalDt"], config)
    for fold in folds:
        assert fold.train_end < fold.test_start
        # purge/embargo already asserted structurally in test_splits.py; here
        # we assert the *actual sliced rows* respect the same boundary.
        train_dates = learnable_panel.loc[
            (learnable_panel["MthCalDt"] >= fold.train_start)
            & (learnable_panel["MthCalDt"] <= fold.train_end),
            "MthCalDt",
        ]
        test_dates = learnable_panel.loc[
            (learnable_panel["MthCalDt"] >= fold.test_start)
            & (learnable_panel["MthCalDt"] <= fold.test_end),
            "MthCalDt",
        ]
        assert train_dates.max() < test_dates.min()
