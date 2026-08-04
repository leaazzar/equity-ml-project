from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_ml.models.baselines import (
    equal_weight_runner,
    fama_macbeth_runner,
    momentum_sort_runner,
)
from equity_ml.models.config import ModelConfig


def _empty(*columns: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def test_equal_weight_runner_scores_every_row_zero() -> None:
    test_df = pd.DataFrame({"f1": [1.0, 2.0, np.nan]})
    preds, params = equal_weight_runner(
        _empty(), _empty(), _empty(), test_df, ["f1"], "target", ModelConfig()
    )
    assert (preds == 0.0).all()
    assert len(preds) == len(test_df)
    assert params == {}


def test_momentum_sort_runner_uses_configured_feature_and_drops_missing() -> None:
    test_df = pd.DataFrame({"mom_12_1": [0.1, np.nan, 0.3]})
    preds, params = momentum_sort_runner(
        _empty(), _empty(), _empty(), test_df, [], "target", ModelConfig()
    )
    assert len(preds) == 2
    assert preds.tolist() == [0.1, 0.3]
    assert params == {"feature": "mom_12_1"}


def test_fama_macbeth_runner_recovers_known_linear_relationship() -> None:
    rng = np.random.default_rng(0)
    months = pd.date_range("2010-01-31", periods=24, freq="ME")
    rows = []
    for month in months:
        x = rng.normal(size=50)
        y = 3.0 * x  # noise-free, exactly recoverable
        for xi, yi in zip(x, y, strict=True):
            rows.append({"MthCalDt": month, "f1": xi, "target": yi})
    train_df = pd.DataFrame(rows)

    test_df = pd.DataFrame({"f1": [1.0, -1.0, 2.0]})
    preds, params = fama_macbeth_runner(
        train_df, _empty(), _empty(), test_df, ["f1"], "target", ModelConfig()
    )
    assert params["n_months_fit"] == 24
    assert preds.tolist() == pytest.approx([3.0, -3.0, 6.0], abs=1e-6)


def test_fama_macbeth_runner_with_no_usable_training_months_returns_nan() -> None:
    train_df = pd.DataFrame({"MthCalDt": [], "f1": [], "target": []})
    test_df = pd.DataFrame({"f1": [1.0, 2.0]})
    preds, params = fama_macbeth_runner(
        train_df, _empty(), _empty(), test_df, ["f1"], "target", ModelConfig()
    )
    assert params["n_months_fit"] == 0
    assert preds.isna().all()
