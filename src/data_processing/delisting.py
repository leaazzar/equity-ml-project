"""Incorporates CRSP delisting returns into the monthly return series.

A security's final monthly observation in `crsp_monthly_stock` does not
include its delisting return — that is reported separately in
`crsp_delisting.DelRet`. Ignoring it understates the true return investors
experienced in the delisting month (a well-known source of survivorship-type
bias in naive CRSP panels — see Shumway (1997)).

Empirically (see MERGE_REPORT.md), a security's last `crsp_monthly_stock`
row is in the same calendar month as its `DelistingDt` for 99.7% of
delisting events, and never more than one month earlier — so the delisting
return is applied to each PERMNO's single last observed row, not matched by
exact calendar date.

Scope: this module only adjusts the delisted security's own final return. It
does not chain returns into a successor security via `DelPERMNO` — that is a
separate, more involved methodology this project has not implemented (see
MERGE_REPORT.md's "not yet handled" section).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def apply_delisting_returns(panel: pd.DataFrame, delisting: pd.DataFrame) -> pd.DataFrame:
    """Add `ret_adj`, `is_last_obs`, `is_delisted`, and `delisting_return_missing` columns.

    `panel` must have `PERMNO`, `MthCalDt`, `MthRet`. `ret_adj` compounds
    `MthRet` with `DelRet` — via `(1 + MthRet) * (1 + DelRet) - 1` — on each
    PERMNO's last observed row if it appears in `delisting`; every other row
    is left as `ret_adj = MthRet`. If `MthRet` itself is null in that final
    row (e.g. no trading occurred before delisting), it is treated as 0 for
    the compounding so `ret_adj` reduces to the delisting return alone.
    Where `DelRet` is null, `ret_adj` for that row is left as the unadjusted
    `MthRet` and `delisting_return_missing` is set `True` rather than
    guessing a proxy return.
    """
    out = panel.copy()
    out["MthCalDt"] = pd.to_datetime(out["MthCalDt"])

    last_obs_idx = out.groupby("PERMNO")["MthCalDt"].idxmax()
    is_last_obs = pd.Series(False, index=out.index)
    is_last_obs.loc[last_obs_idx] = True
    out["is_last_obs"] = is_last_obs

    delisted_permnos = delisting.set_index("PERMNO")["DelRet"]
    is_delisted = out["is_last_obs"] & out["PERMNO"].isin(delisted_permnos.index)
    out["is_delisted"] = is_delisted

    del_ret = out["PERMNO"].map(delisted_permnos).where(is_delisted)
    out["delisting_return_missing"] = is_delisted & del_ret.isna()

    out["ret_adj"] = out["MthRet"]
    has_del_ret = is_delisted & del_ret.notna()
    out.loc[has_del_ret, "ret_adj"] = (1 + out.loc[has_del_ret, "MthRet"].fillna(0)) * (
        1 + del_ret.loc[has_del_ret]
    ) - 1

    n_delisted = int(is_delisted.sum())
    n_missing = int(out["delisting_return_missing"].sum())
    logger.info(
        "Delisting return adjustment: %d PERMNO-months adjusted, %d of those had a "
        "missing DelRet (left unadjusted and flagged, not imputed)",
        n_delisted,
        n_missing,
    )
    return out
