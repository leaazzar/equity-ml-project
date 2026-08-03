"""Investable-universe flag construction.

IMPORTANT LIMITATION: `crsp_monthly_stock` (this project's raw extract) does
not include CRSP's `SHRCD` (share code) or `EXCHCD` (exchange code) fields —
the standard, precise way to identify U.S. common stock (SHRCD 10/11) and
major-exchange listings (EXCHCD 1/2/3 = NYSE/AMEX/NASDAQ). Nor is
`crsp_names` usable here: it has no NAMEDT/NAMEENDDT validity-period columns
(see DATA_DICTIONARY.md), so any share-type/exchange information in it
cannot be applied point-in-time without risking look-ahead bias.

Given that, `is_investable` below is a **coarse, approximate research-
universe proxy** built only from fields actually present and point-in-time-
valid in the master panel: non-missing, positive market cap; a month-end
price at or above a minimum threshold (excludes penny stocks); and a return
flag indicating the security actually traded that month.

**`is_investable` MUST NOT be described, documented, or relied upon as:**
- a verified common-share classification (it does not use `SHRCD` — that
  field does not exist in this extract);
- membership in any named index or benchmark (e.g. "Russell 1000", "S&P
  500", "NYSE/AMEX/NASDAQ common stock") — no such membership is checked or
  implied, and this project has no data source that could verify it
  point-in-time;
- a substitute for a precise, share-code/exchange-code-based investability
  filter used in published academic asset-pricing research.

It is exactly what it is: month-end price ≥ a configurable threshold, a
positive market cap, and evidence of trading that month — nothing more.
Anyone using this flag downstream (feature construction, universe
diagnostics, eventual modeling) must treat it as a rough proxy, not a
validated research universe.

The master panel itself is never filtered by this flag — every row is kept,
and `is_investable` is added as a column so downstream users can subset
explicitly.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.config import FeatureConfig

NON_TRADING_RETURN_FLAG = "NT"


def build_investable_universe(
    panel: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.Series:
    """Return a boolean Series flagging the approximate investable universe.

    A row is flagged `True` when, as of that month: price is non-missing and
    at least `config.min_price_for_universe`; market cap is non-missing and
    positive; and the security actually traded (its return flag is not
    "NT" — CRSP's "no trading" indicator, confirmed in data_validation to
    align exactly with null `MthRet`). Optionally (off by default) excludes
    SIC codes in `config.financial_sic_range`.
    """
    config = config or FeatureConfig()

    has_price = panel["MthPrc"].notna() & (panel["MthPrc"] >= config.min_price_for_universe)
    has_mktcap = panel["MthCap"].notna() & (panel["MthCap"] > 0)
    is_trading = panel["MthRetFlg"] != NON_TRADING_RETURN_FLAG

    universe = has_price & has_mktcap & is_trading

    if config.exclude_financials_from_universe:
        lo, hi = config.financial_sic_range
        is_financial = panel["SICCD"].between(lo, hi)
        universe = universe & ~is_financial

    return universe.fillna(False).astype(bool)
