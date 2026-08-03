"""Tests for MOMENTUM/REVERSAL/VOLATILITY/LIQUIDITY features.

Expected values for rolling windows are computed independently in each test
(via plain `np.prod`/`np.std`/etc. over a hand-picked slice of the same input
series) — this is what actually proves the rolling window is correctly
aligned (the right months, backward-looking, no leakage), rather than just
re-implementing the same formula twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.config import FeatureConfig
from feature_engineering.time_series_features import (
    compute_liquidity_features,
    compute_momentum_reversal_features,
    compute_volatility_features,
)

RET = [
    0.01,
    0.02,
    -0.01,
    0.03,
    0.01,
    -0.02,
    0.02,
    0.01,
    -0.01,
    0.02,
    0.01,
    0.03,
    -0.02,
    0.01,
    0.02,
    -0.01,
    0.03,
    0.01,
    -0.02,
    0.02,
    0.01,
    -0.01,
    0.02,
    0.01,
    0.03,
    -0.02,
    0.01,
    0.02,
    -0.01,
    0.03,
    0.01,
    -0.02,
    0.02,
    0.01,
    -0.01,
    0.02,
    0.01,
    0.03,
    -0.02,
    0.01,
]  # 40 months


def _panel(ret: list[float] = RET, permno: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=len(ret), freq="ME")
    return pd.DataFrame(
        {
            "PERMNO": permno,
            "MthCalDt": dates,
            "ret_adj": ret,
            "ff_mktrf": [0.005] * len(ret),
            "ff_rf": [0.001] * len(ret),
        }
    )


def _cumret(window: list[float]) -> float:
    return float(np.prod([1 + r for r in window]) - 1)


def test_mom_1m_is_current_month_return() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    assert features["mom_1m"].iloc[20] == pytest.approx(RET[20])


def test_mom_12m_matches_independent_cumulative_product() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    t = 25  # 0-indexed row; window is RET[t-11 .. t] inclusive (12 months)
    expected = _cumret(RET[t - 11 : t + 1])
    assert features["mom_12m"].iloc[t] == pytest.approx(expected)


def test_mom_3m_matches_independent_cumulative_product() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    t = 10
    expected = _cumret(RET[t - 2 : t + 1])
    assert features["mom_3m"].iloc[t] == pytest.approx(expected)


def test_mom_12_1_excludes_current_month() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    t = 25
    # window is RET[t-12 .. t-1] — 12 months ending one month before t.
    expected = _cumret(RET[t - 12 : t])
    assert features["mom_12_1"].iloc[t] == pytest.approx(expected)
    # sanity: this must differ from mom_12m (which includes month t).
    assert features["mom_12_1"].iloc[t] != pytest.approx(features["mom_12m"].iloc[t])


def test_mom_6_1_excludes_current_month() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    t = 20
    expected = _cumret(RET[t - 6 : t])
    assert features["mom_6_1"].iloc[t] == pytest.approx(expected)


def test_momentum_nan_until_min_observations_reached() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    # mom_12m needs 12 full months; row 10 (11th row) only has 11 prior+self.
    assert pd.isna(features["mom_12m"].iloc[10])
    assert not pd.isna(features["mom_12m"].iloc[11])


def test_reversal_1m_equals_mom_1m() -> None:
    df = _panel()
    features = compute_momentum_reversal_features(df)
    pd.testing.assert_series_equal(features["reversal_1m"], features["mom_1m"], check_names=False)


def test_momentum_does_not_mix_across_permnos() -> None:
    df1 = _panel(permno=1)
    df2 = _panel(ret=[0.5] * 40, permno=2)
    combined = pd.concat([df1, df2], ignore_index=True)
    features = compute_momentum_reversal_features(combined)

    combined_mom = features["mom_12m"]
    permno2_mask = combined["PERMNO"] == 2
    # PERMNO 2's early (but window-complete) mom_12m must reflect ONLY its
    # own 0.5 returns, not any bleed-over from PERMNO 1's rows.
    t = combined[permno2_mask].index[11]
    assert combined_mom.loc[t] == pytest.approx((1.5**12) - 1)


def test_vol_12m_matches_independent_std() -> None:
    df = _panel()
    features = compute_volatility_features(df)
    t = 20
    expected = float(np.std(RET[t - 11 : t + 1], ddof=1))
    assert features["vol_12m"].iloc[t] == pytest.approx(expected)


def test_downside_vol_uses_only_negative_returns_in_window() -> None:
    df = _panel()
    t = 20
    window = RET[t - 11 : t + 1]
    negatives = [r for r in window if r < 0]
    assert len(negatives) >= 2  # sanity-check the fixture actually has some
    # Lower min_obs so this test isolates "only negatives are used," not the
    # separate min-observations threshold (covered by other tests).
    config = FeatureConfig(volatility_window_months=12, volatility_min_obs=2)
    features = compute_volatility_features(df, config)
    expected = float(np.std(negatives, ddof=1))
    assert features["downside_vol_12m"].iloc[t] == pytest.approx(expected)


def test_max_min_ret_match_independent_window() -> None:
    df = _panel()
    features = compute_volatility_features(df)
    t = 20
    window = RET[t - 11 : t + 1]
    assert features["max_ret_12m"].iloc[t] == pytest.approx(max(window))
    assert features["min_ret_12m"].iloc[t] == pytest.approx(min(window))


def test_beta_and_idio_vol_recover_known_relationship() -> None:
    # Construct excess returns that are an EXACT linear function of mktrf
    # (beta=2, alpha=0, zero noise) so beta_24m must recover 2.0 exactly and
    # idio_vol_24m must be ~0 (up to floating-point noise).
    n = 30
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    mktrf = np.array([0.01 * ((-1) ** i) * (i % 5 + 1) for i in range(n)])
    rf = np.full(n, 0.001)
    excess = 2.0 * mktrf
    ret_adj = excess + rf

    df = pd.DataFrame(
        {"PERMNO": 1, "MthCalDt": dates, "ret_adj": ret_adj, "ff_mktrf": mktrf, "ff_rf": rf}
    )
    features = compute_volatility_features(
        df, FeatureConfig(beta_window_months=24, beta_min_obs=12)
    )
    beta = features["beta_24m"].iloc[-1]
    idio = features["idio_vol_24m"].iloc[-1]
    assert beta == pytest.approx(2.0, abs=1e-6)
    assert idio == pytest.approx(0.0, abs=1e-6)


def test_beta_nan_until_min_observations() -> None:
    n = 30
    dates = pd.date_range("2018-01-31", periods=n, freq="ME")
    df = pd.DataFrame(
        {
            "PERMNO": 1,
            "MthCalDt": dates,
            "ret_adj": np.linspace(-0.05, 0.05, n),
            "ff_mktrf": np.linspace(-0.02, 0.02, n),
            "ff_rf": np.full(n, 0.001),
        }
    )
    config = FeatureConfig(beta_window_months=24, beta_min_obs=12)
    features = compute_volatility_features(df, config)
    assert pd.isna(features["beta_24m"].iloc[10])
    assert not pd.isna(features["beta_24m"].iloc[11])


def test_liquidity_features_denominator_handling() -> None:
    # dollar_volume_millions and shares_outstanding_actual are already
    # unit-normalized (see units.py) — MthVol (actual shares traded) is
    # compared against shares_outstanding_actual (ShrOut converted from
    # thousands to actual shares), not against raw ShrOut directly, which
    # was a real ~1000x turnover bug found during the unit audit.
    df = pd.DataFrame(
        {
            "MthVol": [1000.0, 0.0, 1000.0],
            "shares_outstanding_actual": [10_000.0, 10_000.0, 0.0],
            "dollar_volume_millions": [1.0, 0.0, 1.0],
            "ret_adj": [0.05, 0.05, 0.05],
        }
    )
    features = compute_liquidity_features(df)
    assert features["liquidity_dollar_volume"].iloc[0] == pytest.approx(1.0)
    assert features["liquidity_share_turnover"].iloc[0] == pytest.approx(0.1)
    assert features["liquidity_amihud_illiq"].iloc[0] == pytest.approx(0.05)
    assert pd.isna(features["liquidity_amihud_illiq"].iloc[1])  # zero dollar volume
    assert pd.isna(features["liquidity_share_turnover"].iloc[2])  # zero shares out
    for series in features.values():
        assert not np.isinf(series.astype("float64")).any()


def test_liquidity_share_turnover_unit_regression() -> None:
    """Regression test for the unit bug found during the audit: MthVol is in
    actual shares, ShrOut (hence shares_outstanding_actual) must be too —
    dividing MthVol by raw (thousands-denominated) ShrOut would overstate
    turnover by ~1000x."""
    # 50,000 actual shares traded against 1,000,000 actual shares outstanding
    # (equivalent to ShrOut=1,000 thousands) -> turnover = 0.05 (5%), not 50.
    df = pd.DataFrame(
        {
            "MthVol": [50_000.0],
            "shares_outstanding_actual": [1_000_000.0],
            "dollar_volume_millions": [1.0],
            "ret_adj": [0.01],
        }
    )
    features = compute_liquidity_features(df)
    assert features["liquidity_share_turnover"].iloc[0] == pytest.approx(0.05)
