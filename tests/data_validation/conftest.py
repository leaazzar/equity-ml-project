"""Shared fixtures: tiny synthetic datasets mirroring the real WRDS extracts'
column names, with deliberately embedded data-quality issues so checks can be
tested against exact, known-good expected counts. None of this is real WRDS
data — it exists only to exercise the pipeline's logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def crsp_monthly_df() -> pd.DataFrame:
    rows = [
        # PERMNO 1: two clean months.
        (
            1,
            "AAAAAA10",
            "AAAAAA10",
            "AAA",
            100,
            1000,
            "2020-01-31",
            10.0,
            1000.0,
            0.01,
            0.01,
            "CR",
            100,
            100,
        ),
        (
            1,
            "AAAAAA10",
            "AAAAAA10",
            "AAA",
            100,
            1000,
            "2020-02-29",
            10.1,
            1010.0,
            0.01,
            0.01,
            "CR",
            100,
            100,
        ),
        # PERMNO 2: one row duplicated exactly (exact full-row duplicate).
        (
            2,
            "BBBBBB10",
            "BBBBBB10",
            "BBB",
            200,
            2000,
            "2020-01-31",
            20.0,
            2000.0,
            0.02,
            0.02,
            "CR",
            200,
            100,
        ),
        (
            2,
            "BBBBBB10",
            "BBBBBB10",
            "BBB",
            200,
            2000,
            "2020-01-31",
            20.0,
            2000.0,
            0.02,
            0.02,
            "CR",
            200,
            100,
        ),
        # PERMNO 3: same key (permno+date), differing non-key values (conflicting duplicate).
        (
            3,
            "CCCCCC10",
            "CCCCCC10",
            "CCC",
            300,
            3000,
            "2020-01-31",
            30.0,
            3000.0,
            0.03,
            0.03,
            "CR",
            300,
            100,
        ),
        (
            3,
            "CCCCCC10",
            None,
            None,
            300,
            3000,
            "2020-01-31",
            30.0,
            3000.0,
            0.03,
            0.03,
            "CR",
            300,
            100,
        ),
        # PERMNO 4: last observation well before the overall max date (2020-02-29) -> no
        # delisting record for it -> candidate for "possible missing delisting record".
        (
            4,
            "DDDDDD10",
            "DDDDDD10",
            "DDD",
            400,
            4000,
            "2020-01-31",
            40.0,
            4000.0,
            0.04,
            0.04,
            "CR",
            400,
            100,
        ),
    ]
    columns = [
        "PERMNO",
        "HdrCUSIP",
        "CUSIP",
        "Ticker",
        "PERMCO",
        "SICCD",
        "MthCalDt",
        "MthPrc",
        "MthCap",
        "MthRet",
        "MthRetx",
        "MthRetFlg",
        "MthVol",
        "ShrOut",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def crsp_delisting_df() -> pd.DataFrame:
    rows = [
        (1, "2020-06-15", "MER", "UNAV", 0, 0.01),
        (2, "2020-07-01", "MER", "UNAV", 0, None),  # missing DelRet
        (999, "2020-08-01", "MER", "UNAV", 0, 0.0),  # PERMNO not in crsp_monthly
    ]
    columns = ["PERMNO", "DelistingDt", "DelActionType", "DelReasonType", "DelPERMNO", "DelRet"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def crsp_names_df() -> pd.DataFrame:
    rows = [
        (1, 100, "AAAAAA10", "AAA", "AAA CORP", "EQTY", "NS", 1000, "Q"),
        (1, 100, "AAAAAA10", "AAA", "AAA CORP", "EQTY", "NS", 1000, "Q"),  # exact duplicate
        (2, 200, "BBBBBB10", "BBB", "BBB CORP", "EQTY", "NS", 2000, "N"),
    ]
    columns = [
        "permno",
        "permco",
        "cusip",
        "ticker",
        "issuernm",
        "securitytype",
        "sharetype",
        "siccd",
        "primaryexch",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def ccm_link_table_df() -> pd.DataFrame:
    rows = [
        # gvkey 10: two non-overlapping (adjacent) link periods for the same LIID.
        (10, "AAA", "P", "01", "LU", 1, 100, "2000-01-01", "2010-01-01"),
        (10, "AAA", "P", "01", "LU", 1, 100, "2010-01-02", "E"),
        # gvkey 20: LPERMNO not present in crsp_monthly, but link ends before crsp's
        # earliest date (2020-01-31) -> an "expected" coverage gap.
        (20, "BBB", "P", "01", "LU", 999, 200, "1990-01-01", "1999-12-31"),
        # gvkey 30: overlapping link periods for the same (gvkey, LIID) -> should be flagged.
        (30, "CCC", "P", "01", "LU", 2, 300, "2000-01-01", "2015-01-01"),
        (30, "CCC", "C", "01", "LU", 2, 300, "2010-01-01", "2020-01-01"),
    ]
    columns = [
        "gvkey",
        "tic",
        "LINKPRIM",
        "LIID",
        "LINKTYPE",
        "LPERMNO",
        "LPERMCO",
        "LINKDT",
        "LINKENDDT",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def compustat_df() -> pd.DataFrame:
    rows = [
        (10, "2020-12-31", 2020, "INDL", "C", "D", "STD", "USD"),
        (30, "2020-12-31", 2020, "INDL", "C", "D", "STD", "USD"),
    ]
    columns = ["GVKEY", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "curcd"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def fama_french_df() -> pd.DataFrame:
    rows = [
        ("2020-01-31", 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.01),
        # 2020-02 intentionally omitted: crsp_monthly_df has a 2020-02-29 row, so this
        # creates a real month-coverage mismatch for check_fama_french_date_coverage.
    ]
    columns = ["dateff", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
    return pd.DataFrame(rows, columns=columns)


RAW_CSV_CONTENTS = {
    "crsp_monthly_stock.csv": (
        "PERMNO,HdrCUSIP,CUSIP,Ticker,PERMCO,SICCD,MthCalDt,MthPrc,MthCap,MthRet,MthRetx,MthRetFlg,MthVol,ShrOut\n"
        "1,AAAAAA10,AAAAAA10,AAA,100,1000,2020-01-31,10.0,1000.0,0.01,0.01,CR,100,100\n"
        "1,AAAAAA10,AAAAAA10,AAA,100,1000,2020-02-29,10.1,1010.0,0.01,0.01,CR,100,100\n"
    ),
    "crsp_delisting.csv": (
        "PERMNO,DelistingDt,DelActionType,DelReasonType,DelPERMNO,DelRet\n"
        "1,2020-06-15,MER,UNAV,0,0.01\n"
    ),
    "crsp_names.csv": (
        "permno,permco,cusip,ticker,issuernm,securitytype,sharetype,siccd,primaryexch\n"
        "1,100,AAAAAA10,AAA,AAA CORP,EQTY,NS,1000,Q\n"
    ),
    "ccm_link_table.csv": (
        "gvkey,tic,LINKPRIM,LIID,LINKTYPE,LPERMNO,LPERMCO,LINKDT,LINKENDDT\n"
        "10,AAA,P,01,LU,1,100,2000-01-01,E\n"
    ),
    "compustat_fundamentals_annual.csv": (
        "GVKEY,datadate,fyear,indfmt,consol,popsrc,datafmt,curcd,apdedate\n"
        "10,2020-12-31,2020,INDL,C,D,STD,USD,2021-02-15\n"
    ),
    "fama_french_5f_momentum_monthly.csv": (
        "dateff,mktrf,smb,hml,rmw,cma,rf,umd\n"
        "2020-01-31,0.01,0.01,0.01,0.01,0.01,0.001,0.01\n"
        "2020-02-29,0.02,0.02,0.02,0.02,0.02,0.001,0.02\n"
    ),
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Write a tiny, consistent set of synthetic raw CSVs (all 6 datasets) to tmp_path."""
    d = tmp_path / "raw"
    d.mkdir()
    for filename, content in RAW_CSV_CONTENTS.items():
        (d / filename).write_text(content)
    return d
