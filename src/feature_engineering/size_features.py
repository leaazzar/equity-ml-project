"""SIZE features: market capitalization, log market cap, and relative
market size (a cross-sectional ratio to the investable universe's median).

Uses `mktcap_millions` (from `units.add_unit_normalized_columns`), not the
raw `MthCap` column, so `size_mktcap` is expressed in $ millions — the same
canonical unit as every other monetary feature in this package (`MthCap`
alone is in $ thousands; see `units.py` / `UNIT_AUDIT_REPORT.md`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_size_features(df: pd.DataFrame, is_investable: pd.Series) -> dict[str, pd.Series]:
    """Market cap ($ millions), log market cap, and size relative to that
    month's universe median.

    `relative_mktcap` is still a per-row, interpretable-unit characteristic
    (e.g. "2.3x the median investable-universe firm that month"), computed
    using only the current month's own cross-section — never future months,
    and never the full-sample distribution. It is numerically identical
    whether computed from `MthCap` or `mktcap_millions` (a fixed rescaling
    cancels in a same-unit ratio) — `mktcap_millions` is used here only for
    consistency with `size_mktcap`.
    """
    mktcap = df["mktcap_millions"].where(df["mktcap_millions"] > 0)
    # pandas' nullable-dtype ufunc dispatch computes log() on the underlying
    # raw array (including masked-out <= 0 values) before applying the mask,
    # so numpy warns about log(0)/log(negative) even though the masked
    # result is correctly NaN either way — expected and already masked.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mktcap = np.log(mktcap)

    month = df["MthCalDt"].dt.to_period("M")
    universe_mktcap = mktcap.where(is_investable)
    monthly_median = universe_mktcap.groupby(month).transform("median")
    relative_mktcap = (mktcap / monthly_median).replace([np.inf, -np.inf], np.nan)

    return {
        "size_mktcap": mktcap,
        "size_log_mktcap": log_mktcap,
        "size_relative_mktcap": relative_mktcap,
    }
