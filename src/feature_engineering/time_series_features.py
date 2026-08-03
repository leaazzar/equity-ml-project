"""MOMENTUM, REVERSAL, LIQUIDITY, and VOLATILITY/RISK features.

All rolling computations here are per-PERMNO, backward-looking windows over
`ret_adj` (the delisting-return-adjusted monthly return from
`data_processing`, used throughout rather than raw `MthRet` since it's the
more complete series with no downside). **All windows are in units of
calendar months of the panel's own row cadence**: `data_validation` confirmed
every PERMNO's monthly observations are calendar-contiguous (zero gaps
across all 26,348 PERMNOs as of the last check — see `validation.py`'s
`check_no_calendar_gaps`), so a plain row-position rolling window of size N
is exactly a trailing N-calendar-month window. `min_periods` is always set
equal to the configured minimum observation count, so a window is either
computed on the full required history or left `NaN` — never a partial,
silently-shorter window.

Everything here is monthly-frequency by necessity: this project's CRSP
extract has no daily prices/volumes/returns, so beta, volatility,
max/min return, and Amihud illiquidity are all lower-frequency
approximations of measures more commonly built from daily data. This is a
data-availability limitation, documented per-feature in
FEATURE_DICTIONARY.md, not an oversight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.config import FeatureConfig

NON_TRADING_RETURN_FLAG = "NT"


def _grouped_rolling(
    series: pd.Series, permno: pd.Series, window: int, min_periods: int, op: str
) -> pd.Series:
    """`series.groupby(permno).rolling(window, min_periods).<op>()`, realigned to `series.index`."""
    grouped = getattr(series.groupby(permno).rolling(window, min_periods=min_periods), op)()
    return grouped.reset_index(level=0, drop=True).reindex(series.index)


def _cumulative_return(log_ret: pd.Series, permno: pd.Series, window: int) -> pd.Series:
    rolling_sum = _grouped_rolling(log_ret, permno, window, window, "sum")
    return np.expm1(rolling_sum)


def compute_momentum_reversal_features(
    df: pd.DataFrame, config: FeatureConfig | None = None
) -> dict[str, pd.Series]:
    """1/3/6/9/12-month momentum, 12-1 and 6-1 momentum, and 1-month reversal."""
    config = config or FeatureConfig()
    permno = df["PERMNO"]
    ret = df["ret_adj"]

    # log1p(ret) with ret <= -1 masked out (economically impossible for a
    # legitimate return, but guarded against defensively) so a single bad
    # observation NaNs out every window that contains it, rather than
    # silently producing -inf. `errstate` suppresses the numpy warning from
    # pandas' nullable-dtype ufunc dispatch computing log1p on the masked-out
    # raw values before the mask is applied (same as size_features.py).
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log1p(ret.where(ret > -1))

    features: dict[str, pd.Series] = {"mom_1m": ret}

    for window in config.momentum_windows:
        features[f"mom_{window}m"] = _cumulative_return(log_ret, permno, window)

    # "N-1" momentum: cumulative return over the N months ending one month
    # ago (i.e. excluding the current month) — shift the log-return series
    # forward by one row (within each PERMNO) before applying the same
    # rolling-sum logic used for the plain N-month measures.
    shifted_log_ret = log_ret.groupby(permno).shift(1)
    features["mom_12_1"] = _cumulative_return(shifted_log_ret, permno, 12)
    features["mom_6_1"] = _cumulative_return(shifted_log_ret, permno, 6)

    # Short-term reversal (STR): numerically identical to mom_1m. Included as
    # its own named feature because the literature treats the most recent
    # month's return as a distinct "reversal" signal (expected negative
    # relationship with next-period returns) from intermediate-horizon
    # momentum, even though the underlying statistic is the same number.
    features["reversal_1m"] = ret

    return features


def compute_volatility_features(
    df: pd.DataFrame, config: FeatureConfig | None = None
) -> dict[str, pd.Series]:
    """Rolling volatility, downside volatility, beta, idiosyncratic volatility,
    max/min return, and return skewness — all monthly-frequency."""
    config = config or FeatureConfig()
    permno = df["PERMNO"]
    ret = df["ret_adj"]

    vol = _grouped_rolling(
        ret, permno, config.volatility_window_months, config.volatility_min_obs, "std"
    )

    downside_ret = ret.where(ret < 0)
    downside_vol = _grouped_rolling(
        downside_ret, permno, config.volatility_window_months, config.volatility_min_obs, "std"
    )

    beta, idio_vol = _rolling_beta_and_idio_vol(df, config)

    max_ret = _grouped_rolling(
        ret, permno, config.max_min_window_months, config.max_min_min_obs, "max"
    )
    min_ret = _grouped_rolling(
        ret, permno, config.max_min_window_months, config.max_min_min_obs, "min"
    )

    skew = _grouped_rolling(ret, permno, config.skew_window_months, config.skew_min_obs, "skew")

    return {
        "vol_12m": vol,
        "downside_vol_12m": downside_vol,
        "beta_24m": beta,
        "idio_vol_24m": idio_vol,
        "max_ret_12m": max_ret,
        "min_ret_12m": min_ret,
        "skew_36m": skew,
    }


def _rolling_beta_and_idio_vol(
    df: pd.DataFrame, config: FeatureConfig
) -> tuple[pd.Series, pd.Series]:
    """Rolling CAPM beta and idiosyncratic volatility via the closed-form
    population covariance/variance decomposition:

        beta = Cov(mktrf, excess_ret) / Var(mktrf)
        Var(residual) = Var(excess_ret) - beta^2 * Var(mktrf)

    This is exact for a single-regressor OLS with an intercept (Cov(X, e) = 0
    by construction), computed from rolling means rather than a per-window
    regression loop for tractable performance at panel scale. It uses
    population (N-denominator) moments rather than the small-sample,
    2-parameter degrees-of-freedom adjustment an exact rolling OLS would use
    — a documented, standard simplification for large panels.
    """
    permno = df["PERMNO"]
    x = df["ff_mktrf"]
    y = df["ret_adj"] - df["ff_rf"]
    w, m = config.beta_window_months, config.beta_min_obs

    mean_x = _grouped_rolling(x, permno, w, m, "mean")
    mean_y = _grouped_rolling(y, permno, w, m, "mean")
    mean_xy = _grouped_rolling(x * y, permno, w, m, "mean")
    mean_xx = _grouped_rolling(x * x, permno, w, m, "mean")
    mean_yy = _grouped_rolling(y * y, permno, w, m, "mean")

    var_x = mean_xx - mean_x**2
    var_y = mean_yy - mean_y**2
    cov_xy = mean_xy - mean_x * mean_y

    beta = (cov_xy / var_x.where(var_x > 0)).replace([np.inf, -np.inf], np.nan)
    idio_var = (var_y - beta**2 * var_x).clip(lower=0)
    idio_vol = np.sqrt(idio_var)

    return beta, idio_vol


def compute_liquidity_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Dollar volume ($ millions), share turnover, and a monthly-frequency
    Amihud illiquidity proxy.

    Requires `dollar_volume_millions` and `shares_outstanding_actual`
    (`units.add_unit_normalized_columns`). `MthVol` is in **actual shares**,
    not thousands like `ShrOut` — dividing `MthVol` directly by `ShrOut`
    (without converting `ShrOut` to actual shares first) was a real bug
    found during the unit audit: it overstated turnover by ~1000x. See
    `units.py` / `UNIT_AUDIT_REPORT.md`.

    "Zero-return frequency" (a daily-data-based liquidity measure) is not
    computed: this extract has no daily trading data to count zero-return
    days from — see FEATURE_DICTIONARY.md.
    """
    dollar_volume = df["dollar_volume_millions"]
    share_turnover = (
        df["MthVol"] / df["shares_outstanding_actual"].where(df["shares_outstanding_actual"] > 0)
    ).replace([np.inf, -np.inf], np.nan)
    # Monthly-frequency Amihud (1—2002) proxy: |return| / dollar volume, using
    # this month's own values rather than a daily average — a coarser,
    # noisier approximation than the standard daily-averaged measure, which
    # this project's monthly-only CRSP extract cannot support.
    amihud = (df["ret_adj"].abs() / dollar_volume.where(dollar_volume > 0)).replace(
        [np.inf, -np.inf], np.nan
    )

    return {
        "liquidity_dollar_volume": dollar_volume,
        "liquidity_share_turnover": share_turnover,
        "liquidity_amihud_illiq": amihud,
    }
