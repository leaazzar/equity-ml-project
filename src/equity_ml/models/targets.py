"""Forward-return label construction (MODEL_DESIGN.md Section 1).

Builds a point-in-time-safe forward-looking target from `ret_adj` (the
delisting-adjusted return already computed by `data_processing`), using the
same log-compounding convention `feature_engineering.time_series_features`
uses for trailing momentum, applied forward instead of backward.

No feature in this codebase is ever imputed (FEATURE_DICTIONARY.md); the
same rule applies here — a row's target is NaN whenever fewer than
`horizon_months` future calendar-contiguous observations exist for that
PERMNO, and such rows are dropped downstream rather than filled.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from equity_ml.models.config import ModelConfig

logger = logging.getLogger(__name__)


def _forward_cumulative_return(returns: pd.Series, horizon_months: int) -> pd.Series:
    """Forward `horizon_months`-month compounded return for one PERMNO's
    return series, assumed already sorted chronologically with no calendar
    gaps (guaranteed upstream — see validation.check_no_calendar_gaps in
    feature_engineering, run against the same master-panel return series).

    NaN unless all `horizon_months` future months are non-missing (a window
    is either fully populated or NaN, never partial — the same rule
    time_series_features.py uses for trailing windows).
    """
    log1p_ret = np.log1p(returns)
    log_sum = pd.Series(0.0, index=returns.index)
    all_present = pd.Series(True, index=returns.index)
    for step in range(1, horizon_months + 1):
        shifted = log1p_ret.shift(-step)
        all_present &= shifted.notna()
        log_sum = log_sum.add(shifted.fillna(0.0))
    forward_return = pd.Series(np.expm1(log_sum), index=returns.index)
    forward_return[~all_present] = np.nan
    return forward_return


def build_targets(panel: pd.DataFrame, config: ModelConfig | None = None) -> pd.DataFrame:
    """Build forward-return targets from a point-in-time panel.

    `panel` must contain `config.permno_column`, `config.date_column`,
    `config.return_column`, and `config.universe_column` (e.g.
    `data/processed/master_panel.parquet`, or that panel merged with
    `is_investable` from the feature panels).

    Returns a frame keyed by (permno_column, date_column) with:
    - `target_raw`: the raw forward `horizon_months`-month compounded return.
    - `target_demeaned`: `target_raw` minus that month's cross-sectional mean
      of `target_raw` among investable-universe rows — removes the
      market-wide return component, which a dollar-neutral long-short
      portfolio (MODEL_DESIGN.md Section 6) does not monetize and which is
      not a firm-characteristic-driven signal.
    - `target_valid`: whether `target_raw` is non-missing.
    """
    config = config or ModelConfig()
    required = {
        config.permno_column,
        config.date_column,
        config.return_column,
        config.universe_column,
    }
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel is missing required columns: {sorted(missing)}")

    ordered = panel.sort_values([config.permno_column, config.date_column])
    forward_return = ordered.groupby(config.permno_column, group_keys=False)[
        config.return_column
    ].apply(lambda s: _forward_cumulative_return(s, config.horizon_months))

    targets = ordered[[config.permno_column, config.date_column, config.universe_column]].copy()
    targets["target_raw"] = forward_return
    targets["target_valid"] = targets["target_raw"].notna()

    investable_valid = targets[config.universe_column].fillna(False) & targets["target_valid"]
    month_mean = targets.loc[investable_valid].groupby(config.date_column)["target_raw"].mean()
    targets["target_demeaned"] = targets["target_raw"] - targets[config.date_column].map(month_mean)

    n_valid = int(targets["target_valid"].sum())
    logger.info(
        "Built forward-return targets: %d / %d rows valid (horizon=%d months)",
        n_valid,
        len(targets),
        config.horizon_months,
    )
    return targets.reset_index(drop=True)
