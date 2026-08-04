from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from equity_ml.models.config import ModelConfig
from equity_ml.models.splits import build_walk_forward_folds, slice_by_dates

FULL_HISTORY = pd.Series(pd.date_range("2000-01-31", periods=120, freq="ME"))


def test_folds_are_expanding_not_rolling() -> None:
    folds = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=1),
    )
    assert len(folds) > 1
    for fold in folds:
        assert fold.train_start == FULL_HISTORY.min()


def test_folds_advance_chronologically_with_no_test_overlap() -> None:
    folds = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=24, refit_frequency_months=12, horizon_months=1),
    )
    for prev, nxt in pairwise(folds):
        assert prev.test_end < nxt.test_start
        assert nxt.train_end > prev.train_end


def test_purge_embargo_prevents_train_test_adjacency() -> None:
    horizon = 3
    folds = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=36, refit_frequency_months=12, horizon_months=horizon),
    )
    month_index = {d: i for i, d in enumerate(sorted(FULL_HISTORY.unique()))}
    for fold in folds:
        gap = month_index[pd.Timestamp(fold.test_start)] - month_index[pd.Timestamp(fold.train_end)]
        # A forward-looking `horizon`-month label computed at train_end reaches
        # train_end + horizon; test_start must start strictly after that.
        assert gap > horizon


def test_default_purge_embargo_equals_horizon() -> None:
    folds_h1 = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=36, refit_frequency_months=12, horizon_months=1),
    )
    folds_h6 = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=36, refit_frequency_months=12, horizon_months=6),
    )
    months = sorted(FULL_HISTORY.unique())
    idx = {d: i for i, d in enumerate(months)}
    gap_h1 = idx[pd.Timestamp(folds_h1[0].test_start)] - idx[pd.Timestamp(folds_h1[0].train_end)]
    gap_h6 = idx[pd.Timestamp(folds_h6[0].test_start)] - idx[pd.Timestamp(folds_h6[0].train_end)]
    # gap = 1 (next month) + purge (dropped from train) + embargo (dropped from test)
    assert gap_h1 == 1 + 2 * 1
    assert gap_h6 == 1 + 2 * 6


def test_inner_split_is_within_outer_training_window() -> None:
    folds = build_walk_forward_folds(
        FULL_HISTORY,
        ModelConfig(min_initial_train_months=36, refit_frequency_months=12, horizon_months=1),
    )
    for fold in folds:
        assert fold.train_start <= fold.inner_train_end
        assert fold.inner_train_end < fold.inner_val_start
        assert fold.inner_val_start <= fold.inner_val_end
        assert fold.inner_val_end == fold.train_end
        # The inner split never reaches the fold's own test window.
        assert fold.inner_val_end < fold.test_start


def test_raises_when_initial_window_exceeds_sample() -> None:
    short_history = pd.Series(pd.date_range("2000-01-31", periods=10, freq="ME"))
    with pytest.raises(ValueError):
        build_walk_forward_folds(short_history, ModelConfig(min_initial_train_months=24))


def test_raises_when_window_too_small_for_inner_split_and_purge() -> None:
    with pytest.raises(ValueError):
        build_walk_forward_folds(
            FULL_HISTORY,
            ModelConfig(
                min_initial_train_months=5, inner_validation_fraction=0.5, horizon_months=3
            ),
        )


def test_slice_by_dates_is_inclusive() -> None:
    panel = pd.DataFrame({"MthCalDt": FULL_HISTORY, "value": range(len(FULL_HISTORY))})
    sliced = slice_by_dates(panel, FULL_HISTORY.iloc[0], FULL_HISTORY.iloc[2])
    assert len(sliced) == 3
