from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_ml.models.config import ModelConfig
from equity_ml.models.diagnostics import (
    category_importance,
    compute_ic_series,
    permutation_importance_by_fold,
    summarize_ic,
)
from equity_ml.models.estimators import RIDGE
from equity_ml.models.splits import build_walk_forward_folds


def test_compute_ic_series_one_row_per_model_month() -> None:
    predictions = pd.DataFrame(
        {
            "MthCalDt": [pd.Timestamp("2020-01-31")] * 4 + [pd.Timestamp("2020-02-29")] * 4,
            "model": ["ridge"] * 4 + ["ridge"] * 4,
            "y_true": [1, 2, 3, 4, 1, 2, 3, 4],
            "y_pred": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    ic_series = compute_ic_series(predictions, ModelConfig())
    assert len(ic_series) == 2
    jan = ic_series[ic_series["MthCalDt"] == pd.Timestamp("2020-01-31")].iloc[0]
    feb = ic_series[ic_series["MthCalDt"] == pd.Timestamp("2020-02-29")].iloc[0]
    assert jan["ic"] == pytest.approx(1.0)
    assert feb["ic"] == pytest.approx(-1.0)


def test_summarize_ic_ranks_models_by_mean_ic() -> None:
    ic_series = pd.DataFrame(
        {
            "MthCalDt": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")] * 2,
            "model": ["good", "good", "bad", "bad"],
            "ic": [0.5, 0.7, -0.1, 0.1],
            "n": [10, 10, 10, 10],
        }
    )
    summary = summarize_ic(ic_series)
    assert summary.iloc[0]["model"] == "good"
    assert summary.iloc[0]["ic_mean"] == pytest.approx(0.6)
    assert summary.iloc[0]["n_months"] == 2


def test_permutation_importance_and_category_rollup_end_to_end() -> None:
    rng = np.random.default_rng(0)
    months = pd.date_range("2000-01-31", periods=48, freq="ME")
    rows = []
    for month in months:
        size_mktcap = rng.normal(size=30)
        value_bm = rng.normal(size=30)
        target = 3.0 * size_mktcap + rng.normal(scale=0.1, size=30)
        for i in range(30):
            rows.append(
                {
                    "PERMNO": i,
                    "MthCalDt": month,
                    "size_mktcap": size_mktcap[i],
                    "value_bm": value_bm[i],
                    "target": target[i],
                }
            )
    panel = pd.DataFrame(rows)
    config = ModelConfig(min_initial_train_months=24, refit_frequency_months=24, horizon_months=1)
    folds = build_walk_forward_folds(panel["MthCalDt"], config)

    importances = permutation_importance_by_fold(
        panel, ["size_mktcap", "value_bm"], "target", folds, RIDGE, config
    )
    assert set(importances["feature"]) == {"size_mktcap", "value_bm"}
    size_importance = importances.loc[
        importances["feature"] == "size_mktcap", "importance_mean"
    ].mean()
    bm_importance = importances.loc[importances["feature"] == "value_bm", "importance_mean"].mean()
    assert size_importance > bm_importance  # size_mktcap is the only real signal

    rolled_up = category_importance(importances)
    assert set(rolled_up["category"]) >= {"size", "value"}


def test_category_importance_empty_input() -> None:
    result = category_importance(
        pd.DataFrame(columns=["fold_id", "feature", "importance_mean", "importance_std"])
    )
    assert result.empty
