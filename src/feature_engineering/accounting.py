"""Derived accounting quantities and point-in-time year-over-year growth rates.

Two responsibilities:

1. **Derived single-period quantities** (`add_derived_accounting_columns`) —
   intermediate accounting figures the requested ratio features need but
   that Compustat doesn't provide directly in this extract, most notably a
   **total-assets proxy**. This extract has no `AT` (total assets) field;
   `at_proxy = cst_lt + cst_seq` (total liabilities + total stockholders'
   equity) reconstructs it via the fundamental accounting identity
   (Assets = Liabilities + Equity). This is a documented approximation, not
   the literal reported `AT` figure, and is used everywhere an
   asset-denominated ratio is requested (ROA, asset growth, asset turnover,
   gross/operating profitability, leverage ratios).

2. **Year-over-year growth rates** (`compute_growth_compustat_table`) —
   computed on the actual gvkey/fiscal-year sequence (via `groupby(GVKEY)`
   + `shift(1)`, sorted by `datadate`), *before* the point-in-time monthly
   merge, then attached to the panel by reusing
   `data_processing.compustat_merge.merge_compustat_point_in_time` — the
   same tested backward-`merge_asof` + lag + staleness-expiry logic used for
   every other Compustat field. This is deliberately NOT computed by
   shifting the monthly panel back ~12 months, which would risk misalignment
   whenever a fiscal year's data is still "on the shelf" 12 months later
   (see MERGE_REPORT.md's staleness-expiry discussion) — operating on the
   fiscal-year-level table avoids that pitfall entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_processing.compustat_merge import merge_compustat_point_in_time
from data_processing.config import MergeConfig
from feature_engineering.config import FeatureConfig

# Compustat items whose year-over-year growth rate is requested.
_GROWTH_SOURCE_ITEMS = ("at_proxy", "revt", "capx", "invt", "rect", "ppent", "seq")


def add_derived_accounting_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `at_proxy`, `book_equity`, `total_debt`, `enterprise_value`, `net_debt`.

    Requires `mktcap_millions` to already be present (see
    `units.add_unit_normalized_columns`) — `enterprise_value` combines it
    with Compustat fields, and both must be in the same unit ($ millions;
    `MthCap` alone is in $ thousands — see `units.py` / `UNIT_AUDIT_REPORT.md`).
    Every `cst_*` field used here is already in $ millions natively.
    """
    out = panel.copy()

    out["at_proxy"] = out["cst_lt"] + out["cst_seq"]

    # Book equity, following the standard Fama-French construction simplified
    # to the fields available here: stockholders' equity, minus preferred
    # stock (treated as 0 when missing — the overwhelmingly common case is
    # that a firm simply has none, not that the figure is unknown), plus
    # balance-sheet deferred taxes/investment tax credit. All in $ millions.
    pstk_filled = out["cst_pstk"].fillna(0)
    txditc_filled = out["cst_txditc"].fillna(0)
    out["book_equity"] = out["cst_seq"] - pstk_filled + txditc_filled

    out["total_debt"] = out["cst_dlc"] + out["cst_dltt"]

    # Enterprise value ($ millions) = market equity + total debt + preferred
    # stock - cash. Minority interest is not available in this extract and
    # is omitted (a documented limitation, not an assumption of zero
    # minority interest). Uses mktcap_millions, NOT raw MthCap (which is in
    # $ thousands, not millions) — mixing the two here was a real bug found
    # during the unit audit; see UNIT_AUDIT_REPORT.md.
    out["enterprise_value"] = (
        out["mktcap_millions"] + out["total_debt"] + pstk_filled - out["cst_che"]
    )
    out["net_debt"] = out["total_debt"] - out["cst_che"]

    return out


def _growth_rate(current: pd.Series, prior: pd.Series, config: FeatureConfig) -> pd.Series:
    valid_base = prior > 0 if config.growth_requires_positive_prior else prior != 0
    growth = (current - prior) / prior.abs()
    return growth.where(valid_base)


def compute_growth_compustat_table(
    compustat: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Compute YoY growth rates on the gvkey/fiscal-year sequence.

    Returns a frame with `GVKEY`, `datadate`, and one `<item>_growth` column
    per item in `_GROWTH_SOURCE_ITEMS`, suitable for
    `data_processing.compustat_merge.merge_compustat_point_in_time`.
    """
    config = config or FeatureConfig()

    cst = compustat.copy()
    cst["GVKEY"] = cst["GVKEY"].astype("int64")
    cst["datadate"] = pd.to_datetime(cst["datadate"])
    cst["at_proxy"] = cst["lt"] + cst["seq"]
    cst = cst.sort_values(["GVKEY", "datadate"])

    grouped = cst.groupby("GVKEY")
    for item in _GROWTH_SOURCE_ITEMS:
        prior = grouped[item].shift(1)
        cst[f"{item}_growth"] = _growth_rate(cst[item], prior, config)

    growth_cols = [f"{item}_growth" for item in _GROWTH_SOURCE_ITEMS]
    return cst[["GVKEY", "datadate", *growth_cols]]


def merge_growth_features(
    panel_with_gvkey: pd.DataFrame,
    compustat: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    merge_config: MergeConfig | None = None,
) -> pd.DataFrame:
    """Point-in-time merge YoY growth rates onto the panel.

    Reuses the exact same lag/staleness-expiry logic as the base master
    panel's Compustat merge (`data_processing.compustat_merge`), so a growth
    rate becomes available at the same time as the level data it's derived
    from — never earlier. Assumes the input master panel was built with the
    same `MergeConfig` defaults used here (both default to a 6-month lag /
    12-month shelf life); a mismatch would be a real but undocumented-here
    inconsistency between the base panel and these growth features.
    """
    feature_config = feature_config or FeatureConfig()
    merge_config = merge_config or MergeConfig()

    growth_table = compute_growth_compustat_table(compustat, feature_config)
    # bookkeeping_prefix="cst_growth" (distinct from the value-column prefix,
    # which stays "cst") avoids colliding with the `cst_available_date` /
    # `cst_age_months` / `cst_expired` columns the panel already carries from
    # the base Compustat-*levels* merge in data_processing — reusing the
    # same names here would silently produce duplicate-named columns and
    # corrupt the result (caught by test_pipeline.py's integration tests).
    merged = merge_compustat_point_in_time(
        panel_with_gvkey, growth_table, merge_config, bookkeeping_prefix="cst_growth"
    )

    # Drop the (now distinctly-named) redundant bookkeeping columns this
    # merge recomputes — the panel already carries the base-merge versions.
    return merged.drop(
        columns=["cst_growth_available_date", "cst_growth_age_months", "cst_growth_expired"]
    )


def replace_infinite_with_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace +/-inf with NaN in the given columns (e.g. after a division)."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out
