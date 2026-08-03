"""The formal, machine-readable feature registry.

Every feature produced by this package has exactly one `FeatureSpec` entry
here. `pipeline.py` uses this registry to self-check that every column it
produces is documented, and that every declared source column actually
exists in the input panel (see `validation.check_registry_consistency`).
`FEATURE_DICTIONARY.md` is the prose rendering of this same information —
keep them in sync; `pipeline.py`'s validation pass will fail loudly if a
computed feature has no registry entry.

All entries describe the **raw** characteristic. Every raw feature has a
corresponding column of the same name in the model-ready panel, produced by
the single, uniform month-by-month transform documented in `transforms.py`
(winsorize at configurable percentiles, then cross-sectional rank-scale to
[-1, 1] — both computed within the investable universe of that month only).
There is deliberately no separate registry row per transformed feature: the
transform is identical and mechanical across all 51 features, so duplicating
51 rows into 102 would not add information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's full documentation."""

    name: str
    category: str
    formula: str
    economic_intuition: str
    source_columns: tuple[str, ...]
    lookback_window: str
    min_observations: int | None
    point_in_time_notes: str
    missing_value_treatment: str
    outlier_treatment: str
    layer: str = "raw"
    has_model_ready_version: bool = True
    known_limitations: str = ""
    collinear_with: str | None = None
    """Name of another feature this one is perfectly (or near-perfectly)
    collinear with, if any. Set alongside `exclude_from_model_features`."""
    exclude_from_model_features: bool = False
    """True if this feature should be dropped from any model-training
    feature list (see `model_feature_names()`) — e.g. `reversal_1m`, which
    is numerically identical to `mom_1m` and would otherwise silently
    introduce perfect collinearity into a model. Still present in both
    output layers (raw and model-ready) for its documented, distinct
    economic-interpretation labeling; it is only excluded from the
    *recommended model feature list*, not from the data itself."""


_WINSORIZE_NOTE = (
    "No imputation. In the model-ready layer, winsorized at configurable "
    "percentiles (default 1%/99%) computed within that month's investable "
    "universe only (see transforms.py); the raw layer is never winsorized."
)
_NO_IMPUTATION_NOTE = "Missing source data -> missing feature (NaN). Never imputed."


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # ---------------------------------------------------------------- SIZE
    FeatureSpec(
        name="size_mktcap",
        category="size",
        formula="mktcap_millions = MthCap ($ thousands) / 1,000, masked to NaN if <= 0. Units: $ millions.",
        economic_intuition="Firm size in market-value terms; the classic size (SMB) characteristic.",
        source_columns=("MthCap",),
        lookback_window="contemporaneous (current month)",
        min_observations=None,
        point_in_time_notes="Uses only the current month's own market cap.",
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "MthCap's native unit ($ thousands) was empirically verified, not assumed "
            "— see UNIT_AUDIT_REPORT.md. This feature reports the converted $-millions "
            "value, matching the unit used by every other monetary feature in this package."
        ),
    ),
    FeatureSpec(
        name="size_log_mktcap",
        category="size",
        formula="log(mktcap_millions). Units: log($ millions).",
        economic_intuition="Log-scale size; reduces the influence of extreme mega-caps versus the raw level.",
        source_columns=("MthCap",),
        lookback_window="contemporaneous (current month)",
        min_observations=None,
        point_in_time_notes="Uses only the current month's own market cap.",
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="size_relative_mktcap",
        category="size",
        formula=(
            "mktcap_millions / median(mktcap_millions among that month's investable "
            "universe). Dimensionless (a same-unit ratio; numerically identical "
            "whether computed from mktcap_millions or raw MthCap)."
        ),
        economic_intuition="Size relative to the typical investable firm that month (e.g. 2.3x the median).",
        source_columns=("MthCap", "is_investable", "MthCalDt"),
        lookback_window="contemporaneous (current month's cross-section only)",
        min_observations=None,
        point_in_time_notes=(
            "Cross-sectional reference (the median) is computed using only that "
            "month's investable-universe rows — never other months, never the "
            "full sample."
        ),
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    # --------------------------------------------------------------- VALUE
    FeatureSpec(
        name="value_bm",
        category="value",
        formula=(
            "book_equity ($M) / mktcap_millions ($M), where book_equity = "
            "cst_seq - cst_pstk(fillna 0) + cst_txditc(fillna 0), all $ millions "
            "(Compustat's native unit); NaN if book_equity <= 0. Dimensionless."
        ),
        economic_intuition="Book-to-market; classic HML value characteristic.",
        source_columns=("cst_seq", "cst_pstk", "cst_txditc", "MthCap"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes=(
            "book_equity uses the same 6-month-lagged, 12-month-shelf-life "
            "Compustat merge as every cst_* field; mktcap_millions is contemporaneous. "
            "Updates monthly with the current price, not annually-only as in "
            "the classic Fama-French June-updating convention — a deliberate "
            "simplification for this monthly panel."
        ),
        missing_value_treatment="NaN if book_equity <= 0 (negative book equity makes BM sign-flipped/uninterpretable) or if mktcap_millions missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Numerator and denominator were confirmed to be in different native units "
            "(Compustat $ millions vs. CRSP $ thousands) during a dedicated unit audit "
            "— see UNIT_AUDIT_REPORT.md. mktcap_millions (MthCap / 1,000), not raw "
            "MthCap, is used here; this was a real bug in an earlier version of this feature."
        ),
    ),
    FeatureSpec(
        name="value_earnings_yield",
        category="value",
        formula="cst_ni ($M) / mktcap_millions ($M). Dimensionless.",
        economic_intuition="Earnings yield (E/P); inverse of the P/E ratio.",
        source_columns=("cst_ni", "MthCap"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="cst_ni is lag/expiry-safe; mktcap_millions is contemporaneous.",
        missing_value_treatment="NaN if mktcap_millions missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Uses net income (NI); income before extraordinary items (IB) is an "
            "available alternative not computed separately, to avoid a near-duplicate "
            "feature. Denominator is market capitalization (mktcap_millions = MthCap "
            "/ 1,000), never the raw per-share price alone — see UNIT_AUDIT_REPORT.md."
        ),
    ),
    FeatureSpec(
        name="value_cf_yield",
        category="value",
        formula="cst_oancf ($M) / mktcap_millions ($M). Dimensionless.",
        economic_intuition="Operating cash-flow yield; a value measure less sensitive to accrual manipulation than earnings yield.",
        source_columns=("cst_oancf", "MthCap"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="cst_oancf is lag/expiry-safe; mktcap_millions is contemporaneous.",
        missing_value_treatment="NaN if mktcap_millions missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Denominator is market capitalization (mktcap_millions = MthCap / 1,000), never the raw per-share price alone.",
    ),
    FeatureSpec(
        name="value_sales_to_price",
        category="value",
        formula="cst_revt ($M) / mktcap_millions ($M). Dimensionless.",
        economic_intuition="Sales-to-price; a value measure robust to earnings-management concerns.",
        source_columns=("cst_revt", "MthCap"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="cst_revt is lag/expiry-safe; mktcap_millions is contemporaneous.",
        missing_value_treatment="NaN if mktcap_millions missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Denominator is market capitalization (mktcap_millions = MthCap / 1,000), never the raw per-share price alone.",
    ),
    FeatureSpec(
        name="value_ebitda_to_ev",
        category="value",
        formula=(
            "cst_ebitda ($M) / enterprise_value ($M), where enterprise_value = "
            "mktcap_millions + total_debt + cst_pstk(fillna 0) - cst_che, all $ "
            "millions. Dimensionless."
        ),
        economic_intuition="EBITDA yield on enterprise value; a capital-structure-neutral value measure.",
        source_columns=("cst_ebitda", "MthCap", "cst_dlc", "cst_dltt", "cst_pstk", "cst_che"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="All cst_* inputs are lag/expiry-safe; mktcap_millions is contemporaneous.",
        missing_value_treatment="NaN if enterprise_value missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Enterprise value omits minority interest (not available in this extract). "
            "Uses mktcap_millions (MthCap / 1,000), not raw MthCap — mixing $-thousands "
            "MthCap directly with $-millions Compustat fields in enterprise_value was a "
            "real bug found during a dedicated unit audit; see UNIT_AUDIT_REPORT.md."
        ),
    ),
    FeatureSpec(
        name="value_cf_to_ev",
        category="value",
        formula="cst_oancf ($M) / enterprise_value ($M). Dimensionless.",
        economic_intuition="Operating cash flow yield on enterprise value.",
        source_columns=("cst_oancf", "MthCap", "cst_dlc", "cst_dltt", "cst_pstk", "cst_che"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="All cst_* inputs are lag/expiry-safe; mktcap_millions is contemporaneous.",
        missing_value_treatment="NaN if enterprise_value missing/non-positive.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Enterprise value omits minority interest (not available in this extract). "
            "Uses mktcap_millions, not raw MthCap — see UNIT_AUDIT_REPORT.md."
        ),
    ),
    # ------------------------------------------------------------ MOMENTUM
    FeatureSpec(
        name="mom_1m",
        category="momentum",
        formula="ret_adj (current month)",
        economic_intuition="Most recent month's return.",
        source_columns=("ret_adj",),
        lookback_window="1 month (contemporaneous)",
        min_observations=1,
        point_in_time_notes="Uses only the current month's realized return, known as of that month's end.",
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    *[
        FeatureSpec(
            name=f"mom_{w}m",
            category="momentum",
            formula=f"exp(sum(log(1+ret_adj) over trailing {w} months)) - 1",
            economic_intuition=f"Cumulative return over the trailing {w} months (inclusive of the current month).",
            source_columns=("ret_adj", "PERMNO"),
            lookback_window=f"{w} months, backward-looking",
            min_observations=w,
            point_in_time_notes=f"Rolling window strictly requires {w} full prior (calendar-contiguous) monthly observations; no partial windows.",
            missing_value_treatment=f"NaN unless all {w} months in the window are non-missing.",
            outlier_treatment=_WINSORIZE_NOTE,
        )
        for w in (3, 6, 9, 12)
    ],
    FeatureSpec(
        name="mom_12_1",
        category="momentum",
        formula="exp(sum(log(1+ret_adj) over months t-12..t-1)) - 1",
        economic_intuition="Classic 12-month momentum skipping the most recent month (Jegadeesh-Titman / Fama-French UMD convention).",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="12 months ending one month ago",
        min_observations=12,
        point_in_time_notes="Return series shifted forward one row before the rolling sum, so month t itself is excluded.",
        missing_value_treatment="NaN unless all 12 months in the window are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="mom_6_1",
        category="momentum",
        formula="exp(sum(log(1+ret_adj) over months t-6..t-1)) - 1",
        economic_intuition="6-month momentum skipping the most recent month.",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="6 months ending one month ago",
        min_observations=6,
        point_in_time_notes="Return series shifted forward one row before the rolling sum, so month t itself is excluded.",
        missing_value_treatment="NaN unless all 6 months in the window are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    # ----------------------------------------------------------- REVERSAL
    FeatureSpec(
        name="reversal_1m",
        category="reversal",
        formula="ret_adj (current month)",
        economic_intuition=(
            "Short-term reversal (STR): the most recent month's return, "
            "conventionally treated in the literature as having a negative "
            "relationship with next-period returns — a distinct economic "
            "interpretation from mom_1m even though the computed value is "
            "identical."
        ),
        source_columns=("ret_adj",),
        lookback_window="1 month (contemporaneous)",
        min_observations=1,
        point_in_time_notes="Uses only the current month's realized return.",
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Perfectly collinear with mom_1m (identical values in both output layers). Kept in the data for its documented, distinct economic-interpretation label, but excluded from the recommended model feature list — see model_feature_names().",
        collinear_with="mom_1m",
        exclude_from_model_features=True,
    ),
    # ------------------------------------------------- QUALITY/PROFITABILITY
    FeatureSpec(
        name="quality_roe",
        category="quality",
        formula="cst_ni / cst_seq, NaN if cst_seq <= 0",
        economic_intuition="Return on equity.",
        source_columns=("cst_ni", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_seq missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="quality_roa",
        category="quality",
        formula="cst_ni / at_proxy, where at_proxy = cst_lt + cst_seq; NaN if at_proxy <= 0",
        economic_intuition="Return on assets.",
        source_columns=("cst_ni", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="at_proxy is a derived total-assets proxy (this extract has no AT field) — see accounting.py / DATA_DICTIONARY.md.",
    ),
    FeatureSpec(
        name="quality_gross_profitability",
        category="quality",
        formula="cst_gp / at_proxy, NaN if at_proxy <= 0",
        economic_intuition="Gross profitability (Novy-Marx 2013): a robust quality/profitability characteristic.",
        source_columns=("cst_gp", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="quality_operating_profitability",
        category="quality",
        formula="(cst_revt - cst_cogs.fillna(0) - cst_xsga.fillna(0) - cst_xint.fillna(0)) / book_equity, NaN if book_equity <= 0",
        economic_intuition="Operating profitability (Fama-French 2015 RMW definition).",
        source_columns=(
            "cst_revt",
            "cst_cogs",
            "cst_xsga",
            "cst_xint",
            "cst_seq",
            "cst_pstk",
            "cst_txditc",
        ),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="COGS/XSGA/XINT missing treated as 0 (Fama-French's own construction — most firms genuinely have none, not an unknown value); NaN if REVT missing or book_equity <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="quality_net_profit_margin",
        category="quality",
        formula="cst_ni / cst_revt, NaN if cst_revt <= 0",
        economic_intuition="Net profit margin.",
        source_columns=("cst_ni", "cst_revt"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_revt missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="quality_gross_margin",
        category="quality",
        formula="cst_gp / cst_revt, NaN if cst_revt <= 0",
        economic_intuition="Gross margin.",
        source_columns=("cst_gp", "cst_revt"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_revt missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="quality_asset_turnover",
        category="quality",
        formula="cst_revt / at_proxy, NaN if at_proxy <= 0",
        economic_intuition="Asset turnover; sales generated per dollar of assets.",
        source_columns=("cst_revt", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="quality_cf_profitability",
        category="quality",
        formula="cst_oancf / at_proxy, NaN if at_proxy <= 0",
        economic_intuition="Cash-flow-based profitability (assets-scaled), analogous to ROA but cash- rather than earnings-based.",
        source_columns=("cst_oancf", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="quality_cf_margin",
        category="quality",
        formula="cst_oancf / cst_revt, NaN if cst_revt <= 0",
        economic_intuition="Cash-flow margin, analogous to net profit margin but cash-based.",
        source_columns=("cst_oancf", "cst_revt"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_revt missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="quality_accruals",
        category="quality",
        formula="(cst_ni - cst_oancf) / at_proxy, NaN if at_proxy <= 0",
        economic_intuition="Total accruals (cash-flow-statement / Sloan 1996 method), scaled by assets; high accruals are associated with lower subsequent returns/earnings quality.",
        source_columns=("cst_ni", "cst_oancf", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Uses the simpler cash-flow-statement accrual method (NI - OANCF); "
            "the balance-sheet method (change in non-cash working capital) was "
            "not implemented, to control scope."
        ),
    ),
    # -------------------------------------------------- INVESTMENT/GROWTH
    *[
        FeatureSpec(
            name=f"growth_{label}",
            category="growth",
            formula=f"({item}_t - {item}_(t-1)) / abs({item}_(t-1)), NaN if {item}_(t-1) <= 0",
            economic_intuition=f"Year-over-year growth in {label.replace('_', ' ')}.",
            source_columns=(f"cst_{item}_growth",),
            lookback_window="1 fiscal year (current vs. immediately prior fiscal year)",
            min_observations=2,
            point_in_time_notes=(
                "Computed on the gvkey/fiscal-year sequence via groupby-shift "
                "(never by shifting the monthly panel), then merged in with the "
                "same 6-month lag / 12-month shelf life as the level data — see "
                "accounting.py."
            ),
            missing_value_treatment="NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).",
            outlier_treatment=_WINSORIZE_NOTE,
        )
        for item, label in (
            ("at_proxy", "asset"),
            ("revt", "sales"),
            ("capx", "capx"),
            ("invt", "inventory"),
            ("rect", "receivables"),
            ("ppent", "ppent"),
            ("seq", "equity"),
        )
    ],
    # ------------------------------------------- LEVERAGE/FINANCIAL HEALTH
    FeatureSpec(
        name="leverage_debt_to_assets",
        category="leverage",
        formula="total_debt / at_proxy, where total_debt = cst_dlc + cst_dltt; NaN if at_proxy <= 0",
        economic_intuition="Total debt relative to assets.",
        source_columns=("cst_dlc", "cst_dltt", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="leverage_debt_to_equity",
        category="leverage",
        formula="total_debt / cst_seq, NaN if cst_seq <= 0",
        economic_intuition="Total debt relative to book equity.",
        source_columns=("cst_dlc", "cst_dltt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_seq missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="leverage_lt_debt_ratio",
        category="leverage",
        formula="cst_dltt / at_proxy, NaN if at_proxy <= 0",
        economic_intuition="Long-term debt relative to assets.",
        source_columns=("cst_dltt", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="leverage_current_ratio",
        category="leverage",
        formula="cst_act / cst_lct, NaN if cst_lct <= 0",
        economic_intuition="Current assets relative to current liabilities; short-term liquidity/solvency.",
        source_columns=("cst_act", "cst_lct"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_lct missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="leverage_cash_ratio",
        category="leverage",
        formula="cst_che / cst_lct, NaN if cst_lct <= 0",
        economic_intuition="Cash relative to current liabilities; the strictest short-term solvency measure.",
        source_columns=("cst_che", "cst_lct"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_lct missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    FeatureSpec(
        name="leverage_net_debt_to_assets",
        category="leverage",
        formula="net_debt / at_proxy, where net_debt = total_debt - cst_che; NaN if at_proxy <= 0",
        economic_intuition="Debt net of cash holdings, relative to assets.",
        source_columns=("cst_dlc", "cst_dltt", "cst_che", "cst_lt", "cst_seq"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if at_proxy missing or <= 0 (net_debt itself may legitimately be negative — a net cash position).",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Uses the at_proxy total-assets proxy.",
    ),
    FeatureSpec(
        name="leverage_interest_coverage",
        category="leverage",
        formula="cst_ebit / cst_xint, NaN if cst_xint <= 0",
        economic_intuition="Ability to cover interest expense from operating earnings.",
        source_columns=("cst_ebit", "cst_xint"),
        lookback_window="most recent point-in-time-available fiscal year",
        min_observations=None,
        point_in_time_notes="Lag/expiry-safe Compustat inputs.",
        missing_value_treatment="NaN if cst_xint missing, zero, or negative (a zero-interest-expense firm has an undefined/infinite coverage ratio, not a real signal).",
        outlier_treatment=_WINSORIZE_NOTE,
    ),
    # ------------------------------------------------------------ LIQUIDITY
    FeatureSpec(
        name="liquidity_dollar_volume",
        category="liquidity",
        formula="dollar_volume_millions = (MthPrc [$/share] * MthVol [actual shares]) / 1,000,000. Units: $ millions.",
        economic_intuition="Dollar trading volume; a basic liquidity/size-of-trading measure.",
        source_columns=("MthPrc", "MthVol"),
        lookback_window="contemporaneous (current month)",
        min_observations=None,
        point_in_time_notes="Uses only the current month's own price and volume.",
        missing_value_treatment=_NO_IMPUTATION_NOTE,
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "MthPrc*MthVol is a genuine dollar amount either way (this was never a "
            "unit-mismatch bug, unlike value_bm etc.) — expressed in $ millions here "
            "purely for consistency with every other monetary feature in this package."
        ),
    ),
    FeatureSpec(
        name="liquidity_share_turnover",
        category="liquidity",
        formula=(
            "MthVol [actual shares] / shares_outstanding_actual [actual shares, = "
            "ShrOut * 1,000], NaN if shares_outstanding_actual <= 0. Dimensionless."
        ),
        economic_intuition="Shares traded as a fraction of shares outstanding.",
        source_columns=("MthVol", "ShrOut"),
        lookback_window="contemporaneous (current month)",
        min_observations=None,
        point_in_time_notes="Uses only the current month's own volume and shares outstanding.",
        missing_value_treatment="NaN if shares_outstanding_actual missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "MthVol is in actual shares while ShrOut is in thousands of shares — "
            "dividing MthVol directly by raw ShrOut overstated turnover by ~1000x "
            "in an earlier version of this feature; a dedicated unit audit caught "
            "it (median implied turnover was ~104x/month, which is impossible). "
            "See UNIT_AUDIT_REPORT.md."
        ),
    ),
    FeatureSpec(
        name="liquidity_amihud_illiq",
        category="liquidity",
        formula="abs(ret_adj) / dollar_volume_millions, NaN if dollar_volume_millions <= 0. Units: 1 / $ millions.",
        economic_intuition="Amihud (2002) illiquidity: price impact per dollar traded.",
        source_columns=("ret_adj", "MthPrc", "MthVol"),
        lookback_window="contemporaneous (current month)",
        min_observations=None,
        point_in_time_notes="Uses only the current month's own return and dollar volume.",
        missing_value_treatment="NaN if dollar_volume_millions missing or <= 0.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "This extract has no daily data, so this is a monthly-frequency "
            "single-observation proxy, not the standard measure averaged over "
            "daily observations within the month — materially noisier. Its scale "
            "depends on the $-millions convention for dollar_volume_millions "
            "(changing that convention rescales this feature proportionally)."
        ),
    ),
    # ------------------------------------------------------- VOLATILITY/RISK
    FeatureSpec(
        name="vol_12m",
        category="volatility",
        formula="std(ret_adj over trailing 12 months)",
        economic_intuition="Total return volatility.",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="12 months, backward-looking",
        min_observations=6,
        point_in_time_notes="Rolling std over the trailing 12 calendar-contiguous months, requiring at least 6 non-missing observations.",
        missing_value_treatment="NaN unless at least 6 of the trailing 12 months are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Monthly-frequency volatility, not the more common daily-based measure (no daily data available).",
    ),
    FeatureSpec(
        name="downside_vol_12m",
        category="volatility",
        formula="std(ret_adj where ret_adj < 0, over trailing 12 months)",
        economic_intuition="Downside (semi-deviation) volatility; risk from negative returns only.",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="12 months, backward-looking",
        min_observations=6,
        point_in_time_notes="Rolling std of the negative-return subset within the trailing 12 calendar-contiguous months.",
        missing_value_treatment="NaN unless at least 6 negative-return months are present in the trailing 12-month window.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Monthly-frequency; requiring 6 of 12 months to be negative is a meaningfully strict bar, so this feature is missing more often than vol_12m by construction.",
    ),
    FeatureSpec(
        name="beta_24m",
        category="volatility",
        formula="Cov(ff_mktrf, ret_adj - ff_rf) / Var(ff_mktrf), over trailing 24 months",
        economic_intuition="CAPM market beta.",
        source_columns=("ret_adj", "ff_mktrf", "ff_rf", "PERMNO"),
        lookback_window="24 months, backward-looking",
        min_observations=12,
        point_in_time_notes="Rolling covariance/variance over the trailing 24 calendar-contiguous months, requiring at least 12 non-missing observations.",
        missing_value_treatment="NaN unless at least 12 of the trailing 24 months are non-missing, or if the market factor's variance is zero over that window.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Monthly-frequency beta, computed from population (not small-sample-adjusted) rolling moments — a standard, documented simplification.",
    ),
    FeatureSpec(
        name="idio_vol_24m",
        category="volatility",
        formula="sqrt(Var(ret_adj - ff_rf) - beta_24m^2 * Var(ff_mktrf)), over trailing 24 months",
        economic_intuition="Idiosyncratic (firm-specific) volatility from the same rolling market-model regression as beta_24m.",
        source_columns=("ret_adj", "ff_mktrf", "ff_rf", "PERMNO"),
        lookback_window="24 months, backward-looking",
        min_observations=12,
        point_in_time_notes="Same rolling window and minimum-observation rule as beta_24m; computed from the identical closed-form variance decomposition.",
        missing_value_treatment="NaN under the same conditions as beta_24m.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Same population-moment simplification as beta_24m; negative variance from numerical noise is clipped to zero before the square root.",
    ),
    FeatureSpec(
        name="max_ret_12m",
        category="volatility",
        formula="max(ret_adj over trailing 12 months)",
        economic_intuition="Maximum monthly return over the trailing year (a monthly analogue of the Bali-Cakici-Whitelaw MAX factor, which is normally built from daily returns).",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="12 months, backward-looking",
        min_observations=6,
        point_in_time_notes="Rolling max over the trailing 12 calendar-contiguous months.",
        missing_value_treatment="NaN unless at least 6 of the trailing 12 months are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Monthly-frequency proxy for what the literature usually computes from daily returns.",
    ),
    FeatureSpec(
        name="min_ret_12m",
        category="volatility",
        formula="min(ret_adj over trailing 12 months)",
        economic_intuition="Minimum monthly return over the trailing year — crash/tail-risk exposure.",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="12 months, backward-looking",
        min_observations=6,
        point_in_time_notes="Rolling min over the trailing 12 calendar-contiguous months.",
        missing_value_treatment="NaN unless at least 6 of the trailing 12 months are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations="Monthly-frequency proxy.",
    ),
    FeatureSpec(
        name="skew_36m",
        category="volatility",
        formula="skewness(ret_adj over trailing 36 months)",
        economic_intuition="Return skewness; lottery-like (positively skewed) versus crash-prone (negatively skewed) return profiles.",
        source_columns=("ret_adj", "PERMNO"),
        lookback_window="36 months, backward-looking",
        min_observations=24,
        point_in_time_notes="Rolling skewness over the trailing 36 calendar-contiguous months, requiring at least 24 non-missing observations.",
        missing_value_treatment="NaN unless at least 24 of the trailing 36 months are non-missing.",
        outlier_treatment=_WINSORIZE_NOTE,
        known_limitations=(
            "Skewness estimated from ~24-36 monthly points is statistically "
            "noisy; the relatively high min_observations bar is a deliberate "
            "attempt to keep this 'meaningful,' per the task's own caveat, but "
            "does not eliminate estimation noise."
        ),
    ),
)


def registry_as_records() -> list[dict[str, object]]:
    """The registry as a list of plain dicts, for CSV/JSON serialization."""
    return [asdict(spec) for spec in FEATURE_REGISTRY]


def feature_names() -> list[str]:
    return [spec.name for spec in FEATURE_REGISTRY]


def feature_names_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in FEATURE_REGISTRY:
        out.setdefault(spec.category, []).append(spec.name)
    return out


def model_feature_names() -> list[str]:
    """The recommended feature list for model training: every registered
    feature *except* those flagged `exclude_from_model_features` (currently
    just `reversal_1m`, perfectly collinear with `mom_1m`). Both output
    layers (raw and model-ready) still contain every feature, including the
    excluded ones — this function only defines which columns a model should
    actually be trained on.
    """
    return [spec.name for spec in FEATURE_REGISTRY if not spec.exclude_from_model_features]


def collinear_feature_pairs() -> list[tuple[str, str]]:
    """(feature, collinear_with) pairs for every feature that declares one."""
    return [(spec.name, spec.collinear_with) for spec in FEATURE_REGISTRY if spec.collinear_with]
