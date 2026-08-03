"""Configuration for feature engineering — every tunable assumption lives
here, with its rationale, rather than being hardcoded inside feature logic.
See FEATURE_DICTIONARY.md for how each parameter is used per feature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Tunable assumptions for universe construction, rolling windows, and
    winsorization. Defaults are documented per-parameter below.
    """

    # --- Investable universe (see universe.py for the full rationale) ---
    min_price_for_universe: float = 1.0
    """Minimum month-end price to be considered investable (excludes penny
    stocks). This extract has no SHRCD/EXCHCD fields, so this price/size-based
    approximation stands in for a precise common-stock/exchange filter."""

    exclude_financials_from_universe: bool = False
    """Whether to exclude SIC codes 6000-6999 (finance/insurance/real estate)
    from the universe. Off by default: "common equity" and "non-financial"
    are different concepts, and this project's data does not confirm SIC
    code semantics precisely enough to bake this in unconditionally."""

    financial_sic_range: tuple[int, int] = (6000, 6999)

    # --- Momentum / reversal windows (months) ---
    momentum_windows: tuple[int, ...] = (3, 6, 9, 12)
    """Windows for "plain" cumulative momentum (inclusive of the current
    month). 12-1 and 6-1 (skip-the-most-recent-month) variants are computed
    separately — see time_series_features.py."""

    # --- Volatility / risk windows (months) ---
    volatility_window_months: int = 12
    volatility_min_obs: int = 6
    beta_window_months: int = 24
    beta_min_obs: int = 12
    skew_window_months: int = 36
    skew_min_obs: int = 24
    max_min_window_months: int = 12
    max_min_min_obs: int = 6

    # --- Accounting growth rates (see accounting.py) ---
    growth_requires_positive_prior: bool = True
    """Growth rates are only defined when the prior fiscal year's value is
    strictly positive (a negative or zero base makes a percentage growth
    rate uninterpretable/misleading)."""

    # --- Winsorization / cross-sectional normalization ---
    winsorize_lower_pct: float = 0.01
    winsorize_upper_pct: float = 0.99
    min_cross_section_for_winsorize: int = 10
    """If a month's investable-universe cross-section has fewer than this
    many non-missing observations for a feature, winsorization is skipped
    for that month (percentiles from <10 points are unreliable) and values
    are passed through unwinsorized."""
