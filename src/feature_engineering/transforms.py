"""The model-ready transform: winsorize, then cross-sectional rank-scale,
both computed month-by-month within the investable universe only.

This is the one, uniform, documented transformation applied identically to
every raw feature to produce its model-ready counterpart (see registry.py's
module docstring for why there isn't a separate registry entry per
transformed feature). An alternative z-score transform is also provided
(tested, available for substitution) but is not what builds the saved
model-ready panel — rank-based normalization is the more standard, outlier-
robust choice for cross-sectional characteristic panels.

**Non-investable rows are NaN in the model-ready output.** A cross-sectional
rank relative to the investable universe isn't well-defined for a security
that isn't part of that universe (there's no principled way to interpolate
its position). The raw panel keeps every row regardless — only the
model-ready panel is universe-relative by construction.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.config import FeatureConfig
from feature_engineering.outliers import winsorize_by_month


def rank_transform_by_month(
    values: pd.Series, month: pd.Series, is_investable: pd.Series
) -> pd.Series:
    """Percentile-rank `values` within each month's investable universe, scaled to [-1, 1]."""
    universe_values = values.where(is_investable)
    pct_rank = universe_values.groupby(month).transform(lambda g: g.rank(pct=True))
    return (pct_rank * 2 - 1).where(is_investable)


def zscore_transform_by_month(
    values: pd.Series, month: pd.Series, is_investable: pd.Series
) -> pd.Series:
    """Cross-sectional z-score within each month's investable universe.

    Documented alternative to `rank_transform_by_month`, not used to build
    the saved model-ready panel (rank normalization is more outlier-robust
    for characteristic panels), but implemented and tested for completeness.
    """
    universe_values = values.where(is_investable)
    grouped = universe_values.groupby(month)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    zscore = (universe_values - mean) / std.where(std > 0)
    return zscore.where(is_investable)


def build_model_ready_column(
    raw_values: pd.Series,
    month: pd.Series,
    is_investable: pd.Series,
    config: FeatureConfig | None = None,
) -> pd.Series:
    """Winsorize (month-by-month, universe-only thresholds), then rank-scale to [-1, 1]."""
    config = config or FeatureConfig()
    winsorized = winsorize_by_month(raw_values, month, is_investable, config)
    return rank_transform_by_month(winsorized, month, is_investable)
