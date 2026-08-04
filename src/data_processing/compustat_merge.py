"""Point-in-time merge of Compustat annual fundamentals onto the CRSP panel.

This is the crux of look-ahead-bias prevention: a fiscal year's fundamentals
(`datadate`) are only attached to CRSP months on or after
`datadate + reporting_lag_months` (never earlier), and are treated as too
stale to use once more than `max_fundamentals_age_months` have elapsed since
that availability date — see `MergeConfig` and `MERGE_REPORT.md` for the
rationale behind both defaults.

Implemented as a backward `merge_asof` on the availability date (grouped by
`gvkey`), which for each PERMNO-month picks the most recent fiscal year's
data an investor could have already known about.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_processing.config import MergeConfig

logger = logging.getLogger(__name__)

# Columns dropped from the merged panel: GVKEY is redundant with the `gvkey`
# already attached via CCM linking; indfmt/consol/popsrc/datafmt/curcd are
# single-valued in this extract (see DATA_DICTIONARY.md) and carry no
# information.
_DROPPED_COMPUSTAT_COLUMNS = (
    "GVKEY",
    "indfmt",
    "consol",
    "popsrc",
    "datafmt",
    "curcd",
)


def compustat_value_columns(compustat: pd.DataFrame) -> list[str]:
    """Compustat columns (other than GVKEY/datadate/format columns) to carry into the panel."""
    return [c for c in compustat.columns if c not in (*_DROPPED_COMPUSTAT_COLUMNS, "datadate")]


def _month_ordinal(dates: pd.Series) -> pd.Series:
    """Integer month index (consecutive across years) for exact month-count arithmetic."""
    result: pd.Series = dates.dt.to_period("M").astype("int64")
    return result


def merge_compustat_point_in_time(
    panel_with_gvkey: pd.DataFrame,
    compustat: pd.DataFrame,
    config: MergeConfig | None = None,
    column_prefix: str = "cst",
    bookkeeping_prefix: str | None = None,
) -> pd.DataFrame:
    """Attach point-in-time Compustat fundamentals to a panel with PERMNO/MthCalDt/gvkey.

    `panel_with_gvkey` must have `PERMNO`, `MthCalDt`, and a nullable `gvkey`
    column (see `ccm_linking.resolve_gvkey_for_panel`). Returns the input
    columns plus one `{column_prefix}_`-prefixed column per retained
    Compustat field, `{bookkeeping_prefix}_available_date`,
    `{bookkeeping_prefix}_age_months`, and `{bookkeeping_prefix}_expired`.

    `bookkeeping_prefix` defaults to `column_prefix` when not given. Callers
    that reuse this function a second time on a panel that *already* has
    e.g. `cst_available_date` (from an earlier call — see
    `feature_engineering.accounting.merge_growth_features`) must pass a
    distinct `bookkeeping_prefix`, or the rename below would silently create
    two columns with the same name and corrupt the result.
    """
    config = config or MergeConfig()
    bookkeeping_prefix = bookkeeping_prefix if bookkeeping_prefix is not None else column_prefix
    available_date_col = f"{bookkeeping_prefix}_available_date"
    age_months_col = f"{bookkeeping_prefix}_age_months"
    expired_col = f"{bookkeeping_prefix}_expired"
    value_cols = compustat_value_columns(compustat)

    cst = compustat.copy()
    cst["GVKEY"] = cst["GVKEY"].astype("int64")
    cst["datadate"] = pd.to_datetime(cst["datadate"])
    cst["_available_date"] = cst["datadate"] + pd.DateOffset(months=config.reporting_lag_months)
    cst = cst.sort_values("_available_date")

    panel = panel_with_gvkey.copy()
    panel["_orig_order"] = range(len(panel))
    panel["MthCalDt"] = pd.to_datetime(panel["MthCalDt"])
    has_gvkey = panel["gvkey"].notna()

    matched = panel.loc[has_gvkey].copy()
    matched["gvkey"] = matched["gvkey"].astype("int64")
    matched = matched.sort_values("MthCalDt")

    merged = pd.merge_asof(
        matched,
        cst[["GVKEY", "_available_date", *value_cols]],
        left_on="MthCalDt",
        right_on="_available_date",
        left_by="gvkey",
        right_by="GVKEY",
        direction="backward",
    )
    merged = merged.drop(columns=["GVKEY"])

    matched_fundamentals = merged["_available_date"].notna()
    age_months = pd.Series(np.nan, index=merged.index, dtype="float64")
    age_months.loc[matched_fundamentals] = (
        _month_ordinal(merged.loc[matched_fundamentals, "MthCalDt"])
        - _month_ordinal(merged.loc[matched_fundamentals, "_available_date"])
    ).astype("float64")

    expired = matched_fundamentals & (age_months > config.max_fundamentals_age_months)
    # Null out the actual data values once expired, but keep the diagnostic
    # columns (available date, age) so reporting can distinguish "never
    # matched anything" from "matched, but the data was too stale to use".
    # `.mask()` (rather than assigning a raw scalar across the whole slice)
    # lets each column apply its own dtype-appropriate missing marker (NA,
    # NaN, or NaT) across the mix of nullable Int64/Float64/string/datetime
    # dtypes in `value_cols`.
    for col in value_cols:
        merged[col] = merged[col].mask(expired)

    merged = merged.rename(columns={"_available_date": available_date_col})
    merged = merged.rename(columns={c: f"{column_prefix}_{c}" for c in value_cols})
    merged[age_months_col] = age_months
    merged[expired_col] = expired

    unmatched = panel.loc[~has_gvkey].copy()
    for col in [f"{column_prefix}_{c}" for c in value_cols]:
        unmatched[col] = np.nan
    unmatched[available_date_col] = pd.NaT
    unmatched[age_months_col] = np.nan
    unmatched[expired_col] = False

    result = pd.concat([merged, unmatched], ignore_index=True)
    result = result.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)

    n_expired = int(result[expired_col].sum())
    n_usable = int((result[available_date_col].notna() & ~result[expired_col]).sum())
    logger.info(
        "Compustat point-in-time merge: %d / %d PERMNO-months matched to usable "
        "(non-expired) fundamentals; %d additionally matched but expired past the "
        "%d-month shelf life",
        n_usable,
        len(result),
        n_expired,
        config.max_fundamentals_age_months,
    )
    return result
