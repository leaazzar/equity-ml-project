from __future__ import annotations

import pandas as pd
import pytest

from equity_ml.models.config import ModelConfig
from equity_ml.models.splits import WalkForwardFold, build_walk_forward_folds
from equity_ml.models.validation import (
    check_fold_purge_embargo,
    check_folds_expanding,
    check_no_duplicate_predictions,
    check_predictions_within_fold_test_window,
    run_all_checks,
)

FULL_HISTORY = pd.Series(pd.date_range("2000-01-31", periods=120, freq="ME"))


@pytest.fixture
def folds() -> list[WalkForwardFold]:
    config = ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=1)
    return build_walk_forward_folds(FULL_HISTORY, config)


def test_purge_embargo_check_passes_on_real_folds(folds: list[WalkForwardFold]) -> None:
    result = check_fold_purge_embargo(folds, ModelConfig(horizon_months=1))
    assert result.passed
    assert result.n_affected == 0


def test_purge_embargo_check_fails_when_horizon_understated() -> None:
    # Folds were built for horizon=1, but checking against a claimed
    # horizon=1 with an artificially shrunk fold gap should fail.
    fold = WalkForwardFold(
        fold_id=0,
        train_start=pd.Timestamp("2000-01-31"),
        train_end=pd.Timestamp("2001-12-31"),
        test_start=pd.Timestamp("2002-01-31"),  # only 1 month after train_end
        test_end=pd.Timestamp("2002-12-31"),
        inner_train_end=pd.Timestamp("2001-06-30"),
        inner_val_start=pd.Timestamp("2001-08-31"),
        inner_val_end=pd.Timestamp("2001-12-31"),
    )
    result = check_fold_purge_embargo([fold], ModelConfig(horizon_months=3))
    assert not result.passed
    assert result.n_affected == 1


def test_folds_expanding_check_passes(folds: list[WalkForwardFold]) -> None:
    result = check_folds_expanding(folds)
    assert result.passed


def test_folds_expanding_check_fails_on_rolling_window() -> None:
    fold0 = WalkForwardFold(
        0,
        pd.Timestamp("2000-01-31"),
        pd.Timestamp("2001-12-31"),
        pd.Timestamp("2002-03-31"),
        pd.Timestamp("2002-12-31"),
        pd.Timestamp("2001-06-30"),
        pd.Timestamp("2001-08-31"),
        pd.Timestamp("2001-12-31"),
    )
    fold1 = WalkForwardFold(
        1,
        pd.Timestamp("2001-01-31"),
        pd.Timestamp("2002-12-31"),
        pd.Timestamp("2003-03-31"),
        pd.Timestamp("2003-12-31"),
        pd.Timestamp("2002-06-30"),
        pd.Timestamp("2002-08-31"),
        pd.Timestamp("2002-12-31"),
    )
    result = check_folds_expanding([fold0, fold1])
    assert not result.passed


def test_no_duplicate_predictions_check() -> None:
    config = ModelConfig()
    ok = pd.DataFrame(
        {
            "PERMNO": [1, 2],
            "MthCalDt": [pd.Timestamp("2020-01-31")] * 2,
            "model": ["ridge", "ridge"],
        }
    )
    assert check_no_duplicate_predictions(ok, config).passed

    dupes = pd.DataFrame(
        {
            "PERMNO": [1, 1],
            "MthCalDt": [pd.Timestamp("2020-01-31")] * 2,
            "model": ["ridge", "ridge"],
        }
    )
    result = check_no_duplicate_predictions(dupes, config)
    assert not result.passed
    assert result.n_affected == 1


def test_predictions_within_fold_test_window_check(folds: list[WalkForwardFold]) -> None:
    config = ModelConfig()
    good = pd.DataFrame(
        {
            "PERMNO": [1],
            "MthCalDt": [folds[0].test_start],
            "model": ["ridge"],
            "fold_id": [folds[0].fold_id],
        }
    )
    assert check_predictions_within_fold_test_window(good, folds, config).passed

    bad = pd.DataFrame(
        {
            "PERMNO": [1],
            "MthCalDt": [folds[0].train_start],  # nowhere near fold 0's test window
            "model": ["ridge"],
            "fold_id": [folds[0].fold_id],
        }
    )
    result = check_predictions_within_fold_test_window(bad, folds, config)
    assert not result.passed
    assert result.n_affected == 1


def test_run_all_checks_returns_one_result_per_check(folds: list[WalkForwardFold]) -> None:
    config = ModelConfig()
    predictions = pd.DataFrame(
        {
            "PERMNO": [1],
            "MthCalDt": [folds[0].test_start],
            "model": ["ridge"],
            "fold_id": [folds[0].fold_id],
        }
    )
    results = run_all_checks(predictions, folds, config)
    assert len(results) == 4
    assert all(r.passed for r in results)
