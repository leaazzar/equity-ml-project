"""Empirically-verified units for every CRSP/Compustat monetary and
share-count field this project reads, and the single canonical unit every
monetary quantity is normalized to before it is combined — by ratio or by
sum — with a value from the other source.

**Nothing here is assumed from generic "standard convention" folklore.**
Each unit below was verified empirically against the actual raw extracts in
this project (see `UNIT_AUDIT_REPORT.md` for the full methodology and
results), using three independent cross-checks:

1. CRSP internal consistency: `MthCap / (MthPrc * ShrOut)` is 1.0 to 6
   decimal places across a 20,000-row sample — confirms `MthCap` and
   `ShrOut` share a consistent (if not yet known) scale with `MthPrc`.
2. CRSP vs. Compustat cross-check: `(cst_csho * cst_prcc_f) / MthCap` has a
   median of ~0.001 across 1.7M matched PERMNO-months. Since `cst_prcc_f` is
   unambiguously dollars-per-share, this pins down `cst_csho` as millions of
   shares *and* `MthCap` as thousands of dollars simultaneously (a ratio
   this tight and this close to a round power of ten, at this scale, is not
   plausibly a coincidence of differing fiscal/calendar reference dates).
3. Real-magnitude sanity check: the largest-average-market-cap security in
   the panel resolves (via `crsp_names`, for identification only — never
   used in any computation) to a well-known, easily-recognized large-cap
   technology company whose implied actual-dollar market capitalization
   under this unit assignment matches its well-publicized real-world market
   cap to the correct order of magnitude, for the correct historical period.

**Verified units:**

| Field | Source | Unit |
| --- | --- | --- |
| `MthCap` | CRSP | thousands of USD |
| `ShrOut` | CRSP | thousands of shares |
| `MthPrc` | CRSP | USD per share (unscaled) |
| `MthVol` | CRSP | **actual shares** (not thousands — confirmed separately: `MthVol / ShrOut` implies a median monthly turnover of ~104x under the naive assumption, which is impossible; dividing by 1,000 gives ~0.10 (10%), a plausible monthly turnover) |
| `cst_csho` | Compustat | millions of shares |
| `cst_prcc_f` | Compustat | USD per share (unscaled) |
| All other `cst_*` dollar fields (`revt`, `ni`, `seq`, `oancf`, `ebitda`, `gp`, `cogs`, `xsga`, `dlc`, `dltt`, `che`, `act`, `lct`, `ebit`, `xint`, `pstk`, `txditc`, `capx`, `invt`, `rect`, `ppent`, `lt`) | Compustat | millions of USD (cross-checked via `cst_seq / cst_csho` implying plausible per-share book values only under this assumption — implausible at 1000x either direction) |

**Canonical unit adopted: millions of USD**, matching Compustat's native
convention (fewer fields need converting). Every monetary *raw feature* in
this package — not just the ones combining both sources — is expressed in
millions of USD, so the whole `features_raw.parquet` panel has one
consistent, documented monetary unit throughout, not a mix.
"""

from __future__ import annotations

import pandas as pd

CRSP_MKTCAP_THOUSANDS_PER_MILLION = 1_000
"""MthCap is in $ thousands; divide by this to get $ millions."""

CRSP_SHARES_THOUSANDS_PER_UNIT = 1_000
"""ShrOut is in thousands of shares; multiply by this to get actual shares."""

DOLLARS_PER_MILLION = 1_000_000


def add_unit_normalized_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `mktcap_millions`, `dollar_volume_millions`, and
    `shares_outstanding_actual` — CRSP-sourced monetary/share quantities
    converted to the units used everywhere else in this package.

    Must run before any function that combines a CRSP monetary value with a
    Compustat one (`accounting.add_derived_accounting_columns`,
    `ratios.compute_value_features`) or that combines `MthVol` with
    `ShrOut` (`time_series_features.compute_liquidity_features`).
    """
    out = panel.copy()

    # MthCap ($ thousands) -> $ millions, matching Compustat's native unit.
    out["mktcap_millions"] = out["MthCap"] / CRSP_MKTCAP_THOUSANDS_PER_MILLION

    # MthPrc (USD/share) * MthVol (actual shares) = actual USD traded;
    # divide by 1e6 for consistency with the $-millions convention used
    # throughout the rest of this package.
    out["dollar_volume_millions"] = (out["MthPrc"] * out["MthVol"]) / DOLLARS_PER_MILLION

    # ShrOut (thousands of shares) -> actual shares, so it can be compared
    # directly with MthVol (already in actual shares) for turnover.
    out["shares_outstanding_actual"] = out["ShrOut"] * CRSP_SHARES_THOUSANDS_PER_UNIT

    return out
