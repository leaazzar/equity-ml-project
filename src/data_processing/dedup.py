"""Resolves the "conflicting duplicate" PERMNO+MthCalDt rows left over after
data_validation's exact-duplicate removal.

data_validation only drops *exact* full-row duplicates. In
`crsp_monthly_stock`, 3,244 rows (as of the last validation run) share a
PERMNO+MthCalDt key with another row but differ in other columns — typically
one copy has a populated CUSIP/Ticker and the other has both null. A master
panel needs exactly one row per PERMNO-month, so this module makes that
resolution explicit and logged rather than leaving it to whatever pandas
would do implicitly on merge.

Rule: within each duplicate PERMNO+MthCalDt group, keep the row with the
fewest null values (i.e. the most complete record); ties keep the first row
encountered, so the result is deterministic given the input row order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DedupSummary:
    """What happened when resolving conflicting duplicate keys."""

    rows_in: int
    rows_out: int
    n_duplicate_groups: int
    n_rows_dropped: int


def resolve_duplicate_keys(
    df: pd.DataFrame, key_columns: list[str]
) -> tuple[pd.DataFrame, DedupSummary]:
    """Keep exactly one row per `key_columns` group, preferring the most complete row."""
    rows_in = len(df)
    dup_mask = df.duplicated(subset=key_columns, keep=False)
    n_duplicate_groups = int(df.loc[dup_mask, key_columns].drop_duplicates().shape[0])

    working = df.copy()
    working["_n_nulls"] = working.isna().sum(axis=1)
    working["_orig_order"] = range(len(working))
    working = working.sort_values(["_n_nulls", "_orig_order"])
    resolved = working.drop_duplicates(subset=key_columns, keep="first")
    resolved = resolved.sort_values("_orig_order").drop(columns=["_n_nulls", "_orig_order"])
    resolved = resolved.reset_index(drop=True)

    n_dropped = rows_in - len(resolved)
    if n_dropped:
        logger.info(
            "Resolved %d conflicting duplicate-key group(s) in %s: dropped %d "
            "less-complete row(s) of %d input rows",
            n_duplicate_groups,
            key_columns,
            n_dropped,
            rows_in,
        )

    summary = DedupSummary(
        rows_in=rows_in,
        rows_out=len(resolved),
        n_duplicate_groups=n_duplicate_groups,
        n_rows_dropped=n_dropped,
    )
    return resolved, summary
