"""Shared fixtures: small synthetic panels covering specific point-in-time
scenarios (never real WRDS data). Each fixture's docstring/comments state
exactly what scenario it's designed to exercise.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def crsp_monthly_df() -> pd.DataFrame:
    rows = [
        # PERMNO 1: plain, continuously-linked security (see ccm fixture), no
        # delisting, several months for the compustat lag/expiry tests.
        (1, "AAA", 100, "2020-04-30", 10.0, 0.01),
        (1, "AAA", 100, "2020-05-31", 10.1, 0.01),
        (1, "AAA", 100, "2020-06-30", 10.2, 0.01),
        (1, "AAA", 100, "2021-06-30", 12.0, 0.01),
        # PERMNO 2: linked to a gvkey with only one fiscal year of fundamentals,
        # to test the staleness/expiry cutoff precisely.
        (2, "BBB", 200, "2016-06-30", 20.0, 0.01),
        (2, "BBB", 200, "2017-06-30", 20.1, 0.01),
        (2, "BBB", 200, "2017-07-31", 20.2, 0.01),
        # PERMNO 3: never appears in the CCM link table at all.
        (3, "CCC", 300, "2020-01-31", 30.0, 0.01),
        # PERMNO 4: linked with an open-ended ("still active") CCM link.
        (4, "DDD", 400, "2025-01-31", 40.0, 0.01),
        # PERMNO 5: CCM link ends in 2016; CRSP data continues to 2018 with no
        # active link by then.
        (5, "EEE", 500, "2018-01-31", 50.0, 0.01),
        # PERMNO 6: conflicting duplicate key (same PERMNO+month, one row more
        # complete than the other) for the dedup test.
        (6, "FFF", 600, "2020-01-31", 60.0, 0.01),
        (6, "FFF", 600, "2020-01-31", 60.0, 0.01),
        # PERMNO 7: delisted, with a DelRet present (compounding test).
        (7, "GGG", 700, "2020-03-31", 5.0, 0.05),
        # PERMNO 8: delisted, DelRet missing (flag, no imputation, test).
        (8, "HHH", 800, "2020-03-31", 6.0, 0.02),
        # PERMNO 9: delisted, but its final MthRet is null (no trading before
        # delisting) — ret_adj should reduce to DelRet alone.
        (9, "III", 900, "2020-03-31", 7.0, None),
        # PERMNO 10: never delisted.
        (10, "JJJ", 1000, "2020-03-31", 8.0, 0.03),
    ]
    columns = ["PERMNO", "Ticker", "PERMCO", "MthCalDt", "MthPrc", "MthRet"]
    df = pd.DataFrame(rows, columns=columns)

    # Give PERMNO 6's second (duplicate) row a less-complete CUSIP/Ticker, like
    # the real conflicting-duplicate pattern found in crsp_monthly_stock.
    df["CUSIP"] = df["Ticker"]
    dup_idx = df.index[(df["PERMNO"] == 6)][1]
    df.loc[dup_idx, ["CUSIP", "Ticker"]] = None

    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df["SICCD"] = 1000
    df["MthCap"] = df["MthPrc"] * 100
    df["MthRetx"] = df["MthRet"]
    df["MthRetFlg"] = df["MthRet"].apply(lambda r: "NT" if pd.isna(r) else "CR")
    df["MthVol"] = 1000
    df["ShrOut"] = 100
    return df


@pytest.fixture
def crsp_delisting_df() -> pd.DataFrame:
    rows = [
        (7, "2020-03-15", "MER", "UNAV", 0, 0.10),
        (8, "2020-03-20", "MER", "UNAV", 0, None),
        (9, "2020-03-25", "MER", "UNAV", 0, -0.50),
    ]
    columns = ["PERMNO", "DelistingDt", "DelActionType", "DelReasonType", "DelPERMNO", "DelRet"]
    df = pd.DataFrame(rows, columns=columns)
    df["DelistingDt"] = pd.to_datetime(df["DelistingDt"])
    return df


@pytest.fixture
def ccm_link_table_df() -> pd.DataFrame:
    rows = [
        # PERMNO 1 <-> gvkey 100, open-ended from 2019.
        (100, "P", "01", "LU", 1, "2019-01-01", None),
        # PERMNO 2 <-> gvkey 200, open-ended from 2015.
        (200, "P", "01", "LU", 2, "2015-01-01", None),
        # PERMNO 4 <-> gvkey 400, open-ended from 2018.
        (400, "P", "01", "LU", 4, "2018-01-01", None),
        # PERMNO 5 <-> gvkey 500, link CLOSED at end of 2016 (not open-ended) —
        # by 2018 there is no active link.
        (500, "P", "01", "LU", 5, "2010-01-01", "2016-12-31"),
    ]
    columns = ["gvkey", "LINKPRIM", "LIID", "LINKTYPE", "LPERMNO", "LINKDT", "LINKENDDT"]
    df = pd.DataFrame(rows, columns=columns)
    df["LPERMCO"] = df["LPERMNO"] * 10
    df["tic"] = "TIC"
    df["LINKDT"] = pd.to_datetime(df["LINKDT"])
    df["LINKENDDT"] = pd.to_datetime(df["LINKENDDT"])  # NaT for the open-ended rows
    return df


@pytest.fixture
def compustat_df() -> pd.DataFrame:
    rows = [
        # gvkey 100: fyear 2019 (datadate 2019-12-31) and fyear 2020
        # (datadate 2020-12-31) — with a 6-month lag, available 2020-06-30 and
        # 2021-06-30 respectively.
        (100, "2019-12-31", 2019, 100.0),
        (100, "2020-12-31", 2020, 200.0),
        # gvkey 200: a single fiscal year (2015-12-31) — available 2016-06-30,
        # and (with a 12-month shelf life) expires after age_months > 12.
        (200, "2015-12-31", 2015, 999.0),
    ]
    columns = ["GVKEY", "datadate", "fyear", "revt"]
    df = pd.DataFrame(rows, columns=columns)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["indfmt"] = "INDL"
    df["consol"] = "C"
    df["popsrc"] = "D"
    df["datafmt"] = "STD"
    df["curcd"] = "USD"
    df["apdedate"] = pd.NaT
    df["costat"] = "A"
    return df


@pytest.fixture
def fama_french_df() -> pd.DataFrame:
    months = [
        "2016-06-30",
        "2017-06-30",
        "2017-07-31",
        "2018-01-31",
        "2019-12-31",
        "2020-01-31",
        "2020-03-31",
        "2020-04-30",
        "2020-05-31",
        "2020-06-30",
        "2021-06-30",
        "2025-01-31",
    ]
    df = pd.DataFrame({"dateff": pd.to_datetime(months)})
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        df[col] = 0.01
    return df


@pytest.fixture
def interim_frames(
    crsp_monthly_df, crsp_delisting_df, ccm_link_table_df, compustat_df, fama_french_df
) -> dict[str, pd.DataFrame]:
    return {
        "crsp_monthly_stock": crsp_monthly_df,
        "crsp_delisting": crsp_delisting_df,
        "ccm_link_table": ccm_link_table_df,
        "compustat_fundamentals_annual": compustat_df,
        "fama_french_5f_momentum_monthly": fama_french_df,
    }
