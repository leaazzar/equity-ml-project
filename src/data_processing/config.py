"""Configuration for the point-in-time merge — every tunable assumption lives
here, with its rationale, so nothing is hardcoded silently inside the merge
logic. See MERGE_REPORT.md for the full justification of each default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MergeConfig:
    """Point-in-time merge assumptions.

    reporting_lag_months:
        Months after Compustat's fiscal-period-end date (`datadate`) before
        that fiscal year's fundamentals are treated as knowable to investors.
        Default 6, following the standard academic convention (Fama and
        French, 1992/1993) used when no confirmed filing/announcement date
        is available. This extract has no `rdq`-equivalent field, and
        `apdedate` in compustat_fundamentals_annual has an unconfirmed
        meaning (see DATA_DICTIONARY.md) and is not used for lag purposes.

    max_fundamentals_age_months:
        Maximum months a fiscal year's fundamentals remain "current" after
        becoming available, before being treated as stale/missing rather
        than carried forward indefinitely. Default 12, so the total shelf
        life of one fiscal year's data is
        [datadate + reporting_lag_months, datadate + reporting_lag_months +
        max_fundamentals_age_months) — the classic 6-month-lag /
        12-month-shelf-life annual rebalancing window.

    linkprim_priority:
        Tie-break order for `ccm_link_table.LINKPRIM` when more than one CCM
        link row is valid for the same PERMNO and month (not observed in
        this data as of the last validation run, but handled defensively).
        "P" (primary) is preferred first, per WRDS's own convention.
    """

    reporting_lag_months: int = 6
    max_fundamentals_age_months: int = 12
    linkprim_priority: tuple[str, ...] = ("P", "C", "J", "N")
