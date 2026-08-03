"""Row-wise accounting-ratio features: VALUE, QUALITY/PROFITABILITY,
INVESTMENT/GROWTH, and LEVERAGE/FINANCIAL HEALTH.

Every ratio here is a simple function of columns already present on the
panel (raw CRSP/Compustat fields, the derived quantities from
`accounting.add_derived_accounting_columns`, and the growth-rate columns
from `accounting.merge_growth_features`) — no rolling time-series logic
lives in this module (see `time_series_features.py` for that).

Denominator handling is uniform and explicit throughout: a ratio's
denominator must be strictly positive (or, where noted, merely non-zero) or
the result is `NaN` — never `inf`, and never a value computed against a
sign-flipped or nonsensical base. This is what `_safe_divide` enforces.

**Units:** every VALUE-category ratio below combines a CRSP quantity
(market cap) with a Compustat quantity (book equity, earnings, cash flow,
sales, EBITDA, enterprise value). All of these must be in the same unit
before dividing — this module uses `mktcap_millions` and
`enterprise_value` (both $ millions, from `accounting.py`), never the raw
`MthCap` column (which is $ *thousands* — see `units.py` /
`UNIT_AUDIT_REPORT.md`). QUALITY/LEVERAGE/GROWTH ratios are Compustat-only
(numerator and denominator both $ millions already, so the unit cancels
regardless), and were unaffected by the unit issue found during the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series, require_positive: bool = True
) -> pd.Series:
    """Divide, returning NaN (never inf) wherever the denominator is invalid."""
    denom = (
        denominator.where(denominator > 0)
        if require_positive
        else denominator.where(denominator != 0)
    )
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def compute_value_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Book-to-market, earnings/cash-flow/sales yields, EBITDA- and CF-to-EV.

    All five denominators (`mktcap_millions`, `enterprise_value`) and every
    numerator (`cst_*` fields) are in $ millions — see the module docstring.
    Market *capitalization* is used throughout, never the raw per-share
    price (`MthPrc`) alone.
    """
    # Book-to-market follows the standard convention of treating non-positive
    # book equity as making BM undefined (a firm with negative book equity
    # would otherwise produce a sign-flipped, uninterpretable "value" signal).
    bm = _safe_divide(df["book_equity"].where(df["book_equity"] > 0), df["mktcap_millions"])
    return {
        "value_bm": bm,
        "value_earnings_yield": _safe_divide(df["cst_ni"], df["mktcap_millions"]),
        "value_cf_yield": _safe_divide(df["cst_oancf"], df["mktcap_millions"]),
        "value_sales_to_price": _safe_divide(df["cst_revt"], df["mktcap_millions"]),
        "value_ebitda_to_ev": _safe_divide(df["cst_ebitda"], df["enterprise_value"]),
        "value_cf_to_ev": _safe_divide(df["cst_oancf"], df["enterprise_value"]),
    }


def compute_quality_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """ROE, ROA, (gross/operating/cash-flow) profitability, margins, accruals."""
    # Operating profitability (Fama-French 2015 definition): missing SG&A and
    # interest expense are treated as zero, following FF's own construction —
    # most firms genuinely have no interest expense or report SG&A as nil,
    # rather than the figure being unknown.
    op_profit_numerator = (
        df["cst_revt"]
        - df["cst_cogs"].fillna(0)
        - df["cst_xsga"].fillna(0)
        - df["cst_xint"].fillna(0)
    )
    return {
        "quality_roe": _safe_divide(df["cst_ni"], df["cst_seq"]),
        "quality_roa": _safe_divide(df["cst_ni"], df["at_proxy"]),
        "quality_gross_profitability": _safe_divide(df["cst_gp"], df["at_proxy"]),
        "quality_operating_profitability": _safe_divide(op_profit_numerator, df["book_equity"]),
        "quality_net_profit_margin": _safe_divide(df["cst_ni"], df["cst_revt"]),
        "quality_gross_margin": _safe_divide(df["cst_gp"], df["cst_revt"]),
        "quality_asset_turnover": _safe_divide(df["cst_revt"], df["at_proxy"]),
        "quality_cf_profitability": _safe_divide(df["cst_oancf"], df["at_proxy"]),
        "quality_cf_margin": _safe_divide(df["cst_oancf"], df["cst_revt"]),
        "quality_accruals": _safe_divide(df["cst_ni"] - df["cst_oancf"], df["at_proxy"]),
    }


def compute_investment_growth_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Renames the point-in-time-merged YoY growth columns to friendly names.

    The growth rates themselves are computed in `accounting.py` on the
    gvkey/fiscal-year sequence (never here) — this function only exposes
    them under the requested feature names.
    """
    return {
        "growth_asset": df["cst_at_proxy_growth"],
        "growth_sales": df["cst_revt_growth"],
        "growth_capx": df["cst_capx_growth"],
        "growth_inventory": df["cst_invt_growth"],
        "growth_receivables": df["cst_rect_growth"],
        "growth_ppent": df["cst_ppent_growth"],
        "growth_equity": df["cst_seq_growth"],
    }


def compute_leverage_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Debt/assets/equity ratios, current/cash ratios, net debt, interest coverage."""
    return {
        "leverage_debt_to_assets": _safe_divide(df["total_debt"], df["at_proxy"]),
        "leverage_debt_to_equity": _safe_divide(df["total_debt"], df["cst_seq"]),
        "leverage_lt_debt_ratio": _safe_divide(df["cst_dltt"], df["at_proxy"]),
        "leverage_current_ratio": _safe_divide(df["cst_act"], df["cst_lct"]),
        "leverage_cash_ratio": _safe_divide(df["cst_che"], df["cst_lct"]),
        "leverage_net_debt_to_assets": _safe_divide(
            df["net_debt"], df["at_proxy"], require_positive=True
        ),
        "leverage_interest_coverage": _safe_divide(df["cst_ebit"], df["cst_xint"]),
    }
