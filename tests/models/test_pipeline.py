from __future__ import annotations

import pandas as pd

from equity_ml.models.config import ModelConfig
from equity_ml.models.pipeline import build_training_panel, default_model_runners


def test_build_training_panel_joins_return_and_attaches_target() -> None:
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    master_panel = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "MthCalDt": months,
            "ret_adj": [0.01, 0.02, 0.03],
        }
    )
    model_ready_features = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "MthCalDt": months,
            "is_investable": [True, True, True],
            "size_mktcap": [0.1, 0.2, 0.3],
        }
    )
    config = ModelConfig(horizon_months=1)
    panel = build_training_panel(master_panel, model_ready_features, config)

    assert "target_raw" in panel.columns
    assert "target_demeaned" in panel.columns
    assert "ret_adj" in panel.columns
    # Month 0's target is month 1's return.
    assert (
        panel.loc[panel["MthCalDt"] == months[0], "target_raw"].iloc[0] == pd.Series([0.02]).iloc[0]
    )
    # Last month has no future return within the horizon -> NaN target.
    assert panel.loc[panel["MthCalDt"] == months[-1], "target_raw"].isna().iloc[0]


def test_default_model_runners_includes_baselines_and_ml_estimators() -> None:
    runners = default_model_runners()
    assert {"equal_weight", "momentum_sort", "fama_macbeth"}.issubset(runners.keys())
    assert {"ridge", "lasso", "elastic_net", "random_forest", "hist_gradient_boosting"}.issubset(
        runners.keys()
    )
