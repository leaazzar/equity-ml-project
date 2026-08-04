from __future__ import annotations

import pandas as pd
import pytest

from equity_ml.backtest.config import BacktestConfig
from equity_ml.backtest.portfolio import build_portfolio_weights


def _predictions_for_month(n: int, date: pd.Timestamp, model: str = "ridge") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PERMNO": range(n),
            "MthCalDt": date,
            "model": model,
            "y_pred": [float(i) for i in range(n)],  # monotonically increasing score
        }
    )


def test_equal_weight_long_short_sums_to_dollar_neutral() -> None:
    date = pd.Timestamp("2020-01-31")
    predictions = _predictions_for_month(20, date)
    config = BacktestConfig(decile_count=10, weighting_scheme="equal")
    weights = build_portfolio_weights(predictions, config)

    assert weights["weight"].sum() == pytest.approx(0.0, abs=1e-9)
    long_leg = weights[weights["weight"] > 0]
    short_leg = weights[weights["weight"] < 0]
    assert long_leg["weight"].sum() == pytest.approx(1.0)
    assert short_leg["weight"].sum() == pytest.approx(-1.0)


def test_equal_weight_longs_highest_scores_shorts_lowest() -> None:
    date = pd.Timestamp("2020-01-31")
    predictions = _predictions_for_month(20, date)
    config = BacktestConfig(decile_count=10, weighting_scheme="equal")
    weights = build_portfolio_weights(predictions, config).set_index("PERMNO")["weight"]

    # Top decile = highest 2 scores (PERMNO 18, 19); bottom decile = lowest 2 (0, 1).
    assert weights[19] > 0
    assert weights[18] > 0
    assert weights[0] < 0
    assert weights[1] < 0
    assert 9 not in weights.index or weights[9] == 0


def test_too_few_names_produces_no_holdings() -> None:
    date = pd.Timestamp("2020-01-31")
    predictions = _predictions_for_month(5, date)  # fewer than 2*decile_count
    config = BacktestConfig(decile_count=10)
    weights = build_portfolio_weights(predictions, config)
    assert weights.empty


def test_score_weighting_is_proportional_within_leg() -> None:
    date = pd.Timestamp("2020-01-31")
    predictions = pd.DataFrame(
        {
            "PERMNO": [0, 1, 2, 3],
            "MthCalDt": date,
            "model": "ridge",
            "y_pred": [-10.0, -1.0, 1.0, 10.0],
        }
    )
    # decile_count=2 splits 4 names into a bottom half (short) and top half
    # (long) of 2 names each.
    config = BacktestConfig(decile_count=2, weighting_scheme="score")
    weights = build_portfolio_weights(predictions, config).set_index("PERMNO")["weight"]
    # Long leg: PERMNO 2 (score 1.0), 3 (score 10.0) -> proportional to score, sum to 1.0.
    assert weights[3] == pytest.approx(10.0 / 11.0)
    assert weights[2] == pytest.approx(1.0 / 11.0)
    # Short leg: PERMNO 0 (score -10.0), 1 (score -1.0) -> proportional to |score|, sum to -1.0.
    assert weights[0] == pytest.approx(-10.0 / 11.0)
    assert weights[1] == pytest.approx(-1.0 / 11.0)


def test_multiple_months_and_models_are_independent() -> None:
    jan = pd.Timestamp("2020-01-31")
    feb = pd.Timestamp("2020-02-29")
    predictions = pd.concat(
        [
            _predictions_for_month(20, jan, model="ridge"),
            _predictions_for_month(20, feb, model="ridge"),
            _predictions_for_month(20, jan, model="lasso"),
        ],
        ignore_index=True,
    )
    config = BacktestConfig(decile_count=10)
    weights = build_portfolio_weights(predictions, config)
    assert set(weights["MthCalDt"].unique()) == {jan, feb}
    assert set(weights["model"].unique()) == {"ridge", "lasso"}
