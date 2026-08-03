"""Month-by-month winsorization.

Every threshold is computed from a single month's own cross-section (the
investable universe only, never other months, never the full sample) — this
is what makes winsorization safe to apply without any risk of using
"test-period" information to set thresholds later applied to earlier data.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from feature_engineering.config import FeatureConfig


def winsorize_by_month(
    values: pd.Series,
    month: pd.Series,
    is_investable: pd.Series,
    config: FeatureConfig | None = None,
) -> pd.Series:
    """Clip `values` to each month's [lower_pct, upper_pct] quantiles.

    Quantile thresholds are computed using only the investable-universe
    subset of that month, then applied to clip every row in that month
    (investable or not) — this keeps the full (unfiltered) panel's row count
    intact while still deriving thresholds from an economically meaningful
    reference set. `.transform()` broadcasts each month's scalar bound back
    to every row in that month without a slow row-by-row Python loop.
    """
    config = config or FeatureConfig()
    min_n = config.min_cross_section_for_winsorize

    universe_values = values.where(is_investable)
    grouped = universe_values.groupby(month)

    def _quantile(pct: float) -> Callable[[pd.Series], float]:
        def _fn(group: pd.Series) -> float:
            valid = group.dropna()
            if len(valid) < min_n:
                return float("nan")
            return float(valid.quantile(pct))

        return _fn

    lower = grouped.transform(_quantile(config.winsorize_lower_pct))
    upper = grouped.transform(_quantile(config.winsorize_upper_pct))

    return values.clip(lower=lower, upper=upper)
