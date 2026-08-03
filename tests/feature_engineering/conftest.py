"""Comprehensive synthetic master-panel fixture covering the required test
scenarios: missing accounting fields, zero/negative denominators, sparse
return histories, delisted securities, duplicated rows, boundary dates
around the 6-month accounting lag, extreme outliers, and securities
entering/leaving the investable universe. None of this is real WRDS data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

MONTHS = pd.date_range("2019-01-31", "2020-12-31", freq="ME")
N = len(MONTHS)  # 24 months

_CST_FIELD_NAMES = (
    "cst_lt",
    "cst_seq",
    "cst_pstk",
    "cst_txditc",
    "cst_dlc",
    "cst_dltt",
    "cst_che",
    "cst_ni",
    "cst_oancf",
    "cst_revt",
    "cst_ebitda",
    "cst_gp",
    "cst_cogs",
    "cst_xsga",
    "cst_xint",
    "cst_act",
    "cst_lct",
    "cst_ebit",
)


def _cst_fields(available: bool, **fields: float) -> dict:
    """Compustat-derived fields for one row.

    When `available` is False, **every** cst_* value is null — not just
    `cst_available_date` — matching what the real point-in-time merge
    actually produces before a fiscal year's availability date (merge_asof
    finds no match at all, so none of that row's cst_* columns are
    populated). Populating value fields unconditionally while only toggling
    `cst_available_date` would be an unrealistic fixture that doesn't
    exercise the lag boundary correctly.
    """
    if not available:
        return dict.fromkeys(_CST_FIELD_NAMES)
    missing = [name for name in _CST_FIELD_NAMES if name not in fields]
    if missing:
        raise ValueError(f"missing required cst fields when available=True: {missing}")
    return fields


def _row(
    permno: int,
    gvkey: float,
    mthprc: float = 20.0,
    mthcap: float = 2000.0,
    mthret: float | None = 0.01,
    mthretflg: str = "CR",
    mthvol: float = 1000.0,
    shrout: float = 100.0,
    siccd: int = 2000,
    link_matched: bool = False,
    cst_available_date=pd.NaT,
    cst_expired: bool = False,
    cst_fields: dict | None = None,
    is_delisted: bool = False,
    delisting_return_missing: bool = False,
    ret_adj: float | None = None,
) -> dict:
    return {
        "PERMNO": permno,
        "PERMCO": permno * 10,
        "MthPrc": mthprc,
        "MthCap": mthcap,
        "MthRet": mthret,
        "MthRetx": mthret,
        "MthRetFlg": mthretflg,
        "MthVol": mthvol,
        "ShrOut": shrout,
        "SICCD": siccd,
        "gvkey": gvkey,
        "link_matched": link_matched,
        "cst_available_date": cst_available_date,
        "cst_expired": cst_expired,
        **(cst_fields or dict.fromkeys(_CST_FIELD_NAMES)),
        "is_last_obs": False,
        "is_delisted": is_delisted,
        "delisting_return_missing": delisting_return_missing,
        "ret_adj": ret_adj if ret_adj is not None else mthret,
        "ff_mktrf": 0.005,
        "ff_smb": 0.001,
        "ff_hml": 0.001,
        "ff_rmw": 0.001,
        "ff_cma": 0.001,
        "ff_rf": 0.0005,
        "ff_umd": 0.001,
    }


_LAG_BOUNDARY = pd.Timestamp("2019-06-30")  # fy2018 (datadate 2018-12-31) + 6-month lag


def _permno_1_clean() -> list[dict]:
    """PERMNO 1 / gvkey 100: complete, well-behaved firm across the whole panel."""
    rows = []
    for i, dt in enumerate(MONTHS):
        available = dt >= _LAG_BOUNDARY
        rows.append(
            {
                **_row(
                    1,
                    100.0,
                    mthret=0.01 + 0.001 * i,
                    link_matched=True,
                    cst_available_date=_LAG_BOUNDARY if available else pd.NaT,
                    cst_fields=_cst_fields(
                        available,
                        cst_lt=200.0,
                        cst_seq=100.0,
                        cst_pstk=0.0,
                        cst_txditc=0.0,
                        cst_dlc=10.0,
                        cst_dltt=40.0,
                        cst_che=20.0,
                        cst_ni=15.0,
                        cst_oancf=18.0,
                        cst_revt=300.0,
                        cst_ebitda=50.0,
                        cst_gp=120.0,
                        cst_cogs=180.0,
                        cst_xsga=60.0,
                        cst_xint=5.0,
                        cst_act=150.0,
                        cst_lct=70.0,
                        cst_ebit=45.0,
                    ),
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_2_missing_fields() -> list[dict]:
    """PERMNO 2 / gvkey 200: cst_ni deliberately missing every other month
    (post-availability — a genuinely missing accounting field, not a
    pre-lag absence)."""
    rows = []
    for i, dt in enumerate(MONTHS):
        available = dt >= _LAG_BOUNDARY
        fields = _cst_fields(
            available,
            cst_lt=150.0,
            cst_seq=80.0,
            cst_pstk=0.0,
            cst_txditc=0.0,
            cst_dlc=5.0,
            cst_dltt=30.0,
            cst_che=10.0,
            cst_ni=12.0,
            cst_oancf=14.0,
            cst_revt=200.0,
            cst_ebitda=30.0,
            cst_gp=80.0,
            cst_cogs=120.0,
            cst_xsga=40.0,
            cst_xint=3.0,
            cst_act=90.0,
            cst_lct=50.0,
            cst_ebit=25.0,
        )
        if available and i % 2 == 0:
            fields = {**fields, "cst_ni": None}
        rows.append(
            {
                **_row(
                    2,
                    200.0,
                    mthret=0.005,
                    link_matched=True,
                    cst_available_date=_LAG_BOUNDARY if available else pd.NaT,
                    cst_fields=fields,
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_3_bad_denominators() -> list[dict]:
    """PERMNO 3 / gvkey 300: negative book equity/assets -> ratios must be NaN, not inf/garbage."""
    rows = []
    for dt in MONTHS:
        available = dt >= _LAG_BOUNDARY
        rows.append(
            {
                **_row(
                    3,
                    300.0,
                    mthret=0.0,
                    link_matched=True,
                    cst_available_date=_LAG_BOUNDARY if available else pd.NaT,
                    cst_fields=_cst_fields(
                        available,
                        cst_lt=200.0,
                        cst_seq=-50.0,
                        cst_pstk=0.0,
                        cst_txditc=0.0,  # at_proxy=150>0, seq<0
                        cst_dlc=10.0,
                        cst_dltt=10.0,
                        cst_che=5.0,
                        cst_ni=5.0,
                        cst_oancf=5.0,
                        cst_revt=0.0,
                        cst_ebitda=1.0,  # revt == 0
                        cst_gp=1.0,
                        cst_cogs=1.0,
                        cst_xsga=1.0,
                        cst_xint=0.0,  # xint == 0
                        cst_act=10.0,
                        cst_lct=0.0,
                        cst_ebit=1.0,  # lct == 0
                    ),
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_4_sparse_returns() -> list[dict]:
    """PERMNO 4: every other month has no trading (null return) -> insufficient
    history for windowed features most of the time."""
    rows = []
    for i, dt in enumerate(MONTHS):
        has_return = i % 2 == 0
        rows.append(
            {
                **_row(
                    4,
                    np.nan,
                    mthret=(0.01 if has_return else None),
                    mthretflg=("CR" if has_return else "NT"),
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_5_delisted() -> list[dict]:
    """PERMNO 5: delisted mid-panel; ret_adj differs from MthRet on the final row."""
    delist_month = MONTHS[11]
    rows = []
    for dt in MONTHS:
        if dt > delist_month:
            continue
        is_final = dt == delist_month
        mthret = 0.02
        rows.append(
            {
                **_row(
                    5,
                    np.nan,
                    mthret=mthret,
                    is_delisted=is_final,
                    ret_adj=(-0.40 if is_final else mthret),  # compounded delisting return
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_7_lag_boundary() -> list[dict]:
    """PERMNO 7 / gvkey 700: fundamentals available exactly at the 6-month
    boundary, never one month earlier."""
    rows = []
    for dt in MONTHS:
        available = dt >= _LAG_BOUNDARY
        rows.append(
            {
                **_row(
                    7,
                    700.0,
                    mthret=0.0,
                    link_matched=True,
                    cst_available_date=_LAG_BOUNDARY if available else pd.NaT,
                    cst_fields=_cst_fields(
                        available,
                        cst_lt=100.0,
                        cst_seq=50.0,
                        cst_pstk=0.0,
                        cst_txditc=0.0,
                        cst_dlc=5.0,
                        cst_dltt=15.0,
                        cst_che=10.0,
                        cst_ni=5.0,
                        cst_oancf=6.0,
                        cst_revt=100.0,
                        cst_ebitda=15.0,
                        cst_gp=40.0,
                        cst_cogs=60.0,
                        cst_xsga=20.0,
                        cst_xint=1.0,
                        cst_act=50.0,
                        cst_lct=25.0,
                        cst_ebit=14.0,
                    ),
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_8_extreme_outlier() -> list[dict]:
    """PERMNO 8 / gvkey 800: one month with an absurdly extreme ROE."""
    rows = []
    for i, dt in enumerate(MONTHS):
        available = dt >= _LAG_BOUNDARY
        extreme = available and i == 15
        rows.append(
            {
                **_row(
                    8,
                    800.0,
                    mthret=0.0,
                    link_matched=True,
                    cst_available_date=_LAG_BOUNDARY if available else pd.NaT,
                    cst_fields=_cst_fields(
                        available,
                        cst_lt=100.0,
                        cst_seq=(0.01 if extreme else 50.0),
                        cst_pstk=0.0,
                        cst_txditc=0.0,
                        cst_dlc=5.0,
                        cst_dltt=15.0,
                        cst_che=10.0,
                        cst_ni=(1_000_000.0 if extreme else 5.0),
                        cst_oancf=6.0,
                        cst_revt=100.0,
                        cst_ebitda=15.0,
                        cst_gp=40.0,
                        cst_cogs=60.0,
                        cst_xsga=20.0,
                        cst_xint=1.0,
                        cst_act=50.0,
                        cst_lct=25.0,
                        cst_ebit=14.0,
                    ),
                ),
                "MthCalDt": dt,
            }
        )
    return rows


def _permno_9_universe_entry_exit() -> list[dict]:
    """PERMNO 9: price dips below the investable-universe threshold for a
    stretch of months, then recovers."""
    rows = []
    for i, dt in enumerate(MONTHS):
        below_threshold = 8 <= i <= 12
        rows.append(
            {
                **_row(9, np.nan, mthprc=(0.5 if below_threshold else 15.0), mthret=0.0),
                "MthCalDt": dt,
            }
        )
    return rows


@pytest.fixture
def synthetic_master_panel() -> pd.DataFrame:
    rows = (
        _permno_1_clean()
        + _permno_2_missing_fields()
        + _permno_3_bad_denominators()
        + _permno_4_sparse_returns()
        + _permno_5_delisted()
        + _permno_7_lag_boundary()
        + _permno_8_extreme_outlier()
        + _permno_9_universe_entry_exit()
    )
    df = pd.DataFrame(rows)
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    return df.reset_index(drop=True)


@pytest.fixture
def synthetic_compustat_interim() -> pd.DataFrame:
    """Minimal Compustat-like table (only the columns growth-rate
    construction needs) for gvkeys 100, 200, 300, 700, 800 across two fiscal
    years, so YoY growth features have something to compute against."""
    rows = []
    for gvkey, base in ((100, 100.0), (200, 80.0), (300, 60.0), (700, 50.0), (800, 50.0)):
        for fy, datadate in ((2018, "2018-12-31"), (2019, "2019-12-31")):
            growth_factor = 1.0 if fy == 2018 else 1.1
            rows.append(
                {
                    "GVKEY": gvkey,
                    "datadate": pd.Timestamp(datadate),
                    "lt": 100.0 * growth_factor,
                    "seq": base * growth_factor,
                    "revt": 200.0 * growth_factor,
                    "capx": 10.0 * growth_factor,
                    "invt": 20.0 * growth_factor,
                    "rect": 15.0 * growth_factor,
                    "ppent": 40.0 * growth_factor,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def duplicated_master_panel(synthetic_master_panel: pd.DataFrame) -> pd.DataFrame:
    """The same panel with one deliberately duplicated PERMNO-month row."""
    dup_row = synthetic_master_panel[synthetic_master_panel["PERMNO"] == 1].iloc[[0]]
    return pd.concat([synthetic_master_panel, dup_row], ignore_index=True)
