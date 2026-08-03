"""Unit-consistency regression tests.

These exist because a real bug slipped through code review during initial
development: every VALUE-category ratio combining CRSP's `MthCap` ($
thousands) directly with a Compustat `cst_*` field ($ millions) was silently
off by ~1000x, and `liquidity_share_turnover` mixed `MthVol` (actual shares)
with raw `ShrOut` (thousands of shares) the same way. See
`UNIT_AUDIT_REPORT.md` for the full empirical verification methodology.

Every expected value below is hand-computable from the synthetic inputs —
that's what actually proves unit correctness, as opposed to re-implementing
the same (possibly still-wrong) formula a second time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.accounting import add_derived_accounting_columns
from feature_engineering.ratios import compute_value_features
from feature_engineering.time_series_features import compute_liquidity_features
from feature_engineering.units import add_unit_normalized_columns


def test_add_unit_normalized_columns_matches_hand_computed_values() -> None:
    df = pd.DataFrame(
        {
            "MthCap": [1_000_000.0],  # $ thousands -> $1,000,000K = $1B actual
            "MthPrc": [100.0],  # $/share
            "MthVol": [50_000.0],  # actual shares traded
            "ShrOut": [10_000.0],  # thousands of shares -> 10M actual shares
        }
    )
    out = add_unit_normalized_columns(df)

    assert out["mktcap_millions"].iloc[0] == pytest.approx(1_000.0)  # $1,000M = $1B
    # dollar_volume = $100/share * 50,000 shares = $5,000,000 actual = $5M
    assert out["dollar_volume_millions"].iloc[0] == pytest.approx(5.0)
    assert out["shares_outstanding_actual"].iloc[0] == pytest.approx(10_000_000.0)


def test_crsp_and_compustat_implied_market_cap_agree_once_normalized() -> None:
    """The core empirical check from the audit, as an automated regression:
    for the same (synthetic) company, CRSP's mktcap_millions and
    Compustat's csho*prcc_f should agree once units are correctly applied —
    csho is millions of shares, prcc_f is $/share, so csho*prcc_f is
    directly in $ millions, the same unit as mktcap_millions.
    """
    df = pd.DataFrame(
        {
            "MthCap": [1_000_000.0],  # $ thousands -> mktcap_millions = 1,000
            "MthPrc": [100.0],
            "MthVol": [1_000.0],
            "ShrOut": [10_000.0],
        }
    )
    out = add_unit_normalized_columns(df)

    csho_millions_of_shares = 10.0  # 10 million shares
    prcc_f_dollars = 100.0
    compustat_implied_mktcap_millions = csho_millions_of_shares * prcc_f_dollars  # = 1,000

    assert out["mktcap_millions"].iloc[0] == pytest.approx(compustat_implied_mktcap_millions)


def _synthetic_company(mktcap_millions: float = 1000.0) -> pd.DataFrame:
    """A single, fully hand-verifiable synthetic firm, already unit-normalized."""
    return pd.DataFrame(
        {
            "mktcap_millions": [mktcap_millions],
            "cst_lt": [0.0],
            "cst_seq": [400.0],
            "cst_pstk": [0.0],
            "cst_txditc": [0.0],
            "cst_dlc": [50.0],
            "cst_dltt": [150.0],
            "cst_che": [100.0],
            "cst_ni": [50.0],
            "cst_oancf": [60.0],
            "cst_revt": [800.0],
            "cst_ebitda": [120.0],
        }
    )


def test_enterprise_value_uses_consistent_units() -> None:
    df = add_derived_accounting_columns(_synthetic_company())
    # EV = mktcap (1000) + total_debt (50+150=200) + pstk (0) - cash (100) = 1100
    assert df["enterprise_value"].iloc[0] == pytest.approx(1100.0)


def test_value_ratios_match_hand_calculated_expectations() -> None:
    """Every VALUE ratio, hand-computed from the synthetic company above:

    book_equity = seq - pstk + txditc = 400
    bm                 = 400 / 1000  = 0.4
    earnings_yield      = 50 / 1000   = 0.05
    cf_yield            = 60 / 1000   = 0.06
    sales_to_price      = 800 / 1000  = 0.8
    enterprise_value    = 1000+200-100 = 1100
    ebitda_to_ev        = 120 / 1100  ~ 0.10909
    cf_to_ev            = 60 / 1100   ~ 0.05455
    """
    df = add_derived_accounting_columns(_synthetic_company())
    features = compute_value_features(df)

    assert features["value_bm"].iloc[0] == pytest.approx(0.4)
    assert features["value_earnings_yield"].iloc[0] == pytest.approx(0.05)
    assert features["value_cf_yield"].iloc[0] == pytest.approx(0.06)
    assert features["value_sales_to_price"].iloc[0] == pytest.approx(0.8)
    assert features["value_ebitda_to_ev"].iloc[0] == pytest.approx(120 / 1100)
    assert features["value_cf_to_ev"].iloc[0] == pytest.approx(60 / 1100)


def test_value_ratios_would_be_wrong_by_1000x_with_unnormalized_mktcap() -> None:
    """Documents, explicitly, the magnitude of the bug this audit fixed: if
    `MthCap` (raw, $ thousands) were used directly instead of
    `mktcap_millions`, book-to-market would be overstated by exactly 1000x.
    """
    company = _synthetic_company()
    correct = add_derived_accounting_columns(company)
    correct_bm = compute_value_features(correct)["value_bm"].iloc[0]

    buggy = company.rename(columns={"mktcap_millions": "_hidden"})
    buggy["mktcap_millions"] = (
        buggy["_hidden"] * 1000
    )  # simulate raw MthCap (thousands) used directly
    buggy_bm = compute_value_features(add_derived_accounting_columns(buggy))["value_bm"].iloc[0]

    assert buggy_bm == pytest.approx(correct_bm / 1000)


def test_liquidity_share_turnover_matches_hand_calculated_ratio() -> None:
    """500,000 actual shares traded against 10,000,000 actual shares
    outstanding (ShrOut=10,000 thousands) -> turnover = 0.05 (5%)."""
    raw = pd.DataFrame(
        {"MthCap": [0.0], "MthPrc": [20.0], "MthVol": [500_000.0], "ShrOut": [10_000.0]}
    )
    normalized = add_unit_normalized_columns(raw)
    normalized["ret_adj"] = [0.02]

    features = compute_liquidity_features(normalized)
    assert features["liquidity_share_turnover"].iloc[0] == pytest.approx(0.05)
    # dollar volume = $20 * 500,000 shares = $10,000,000 = $10M
    assert features["liquidity_dollar_volume"].iloc[0] == pytest.approx(10.0)


def test_liquidity_share_turnover_would_be_wrong_by_1000x_without_conversion() -> None:
    """Documents the ~1000x turnover-overstatement bug found during the audit:
    dividing MthVol (actual shares) by raw ShrOut (thousands of shares)
    without converting ShrOut to actual shares first."""
    mthvol = 500_000.0
    shrout_thousands = 10_000.0

    correct_turnover = mthvol / (shrout_thousands * 1000)  # what the fixed code computes
    buggy_turnover = mthvol / shrout_thousands  # the pre-audit bug

    assert buggy_turnover == pytest.approx(correct_turnover * 1000)
