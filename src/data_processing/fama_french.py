"""Merges Fama-French factors onto the panel by calendar month.

No point-in-time logic is needed here: `fama_french_5f_momentum_monthly` has
no firm-level identifiers, and its `dateff` month-end dates match
`crsp_monthly_stock.MthCalDt` exactly (verified 1:1 in both directions during
data validation — see DATA_DICTIONARY.md), so this is a plain month-key merge.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_FACTOR_COLUMNS = ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")


def merge_fama_french(panel: pd.DataFrame, fama_french: pd.DataFrame) -> pd.DataFrame:
    """Left-join Fama-French factors onto `panel` by month, prefixing columns `ff_`."""
    ff = fama_french.copy()
    ff["_month"] = pd.to_datetime(ff["dateff"]).dt.to_period("M")
    ff = ff[["_month", *_FACTOR_COLUMNS]].rename(columns={c: f"ff_{c}" for c in _FACTOR_COLUMNS})

    out = panel.copy()
    out["_month"] = pd.to_datetime(out["MthCalDt"]).dt.to_period("M")
    merged = out.merge(ff, on="_month", how="left").drop(columns=["_month"])

    n_unmatched = int(merged["ff_mktrf"].isna().sum())
    if n_unmatched:
        logger.warning(
            "%d / %d panel rows have no matching Fama-French month — "
            "unexpected given prior full date-coverage validation",
            n_unmatched,
            len(merged),
        )
    else:
        logger.info("Fama-French merge: all %d panel rows matched a factor month", len(merged))
    return merged
