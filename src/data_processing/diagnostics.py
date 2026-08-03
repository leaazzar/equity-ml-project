"""Computes merge-quality diagnostics for the master panel: coverage,
unmatched observations, missing accounting variables, duplicate-identifier
checks, and firm coverage over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_CST_PREFIX = "cst_"
_NON_VALUE_CST_COLUMNS = {"cst_available_date", "cst_age_months", "cst_expired"}


def _cst_value_columns(panel: pd.DataFrame) -> list[str]:
    return [
        c for c in panel.columns if c.startswith(_CST_PREFIX) and c not in _NON_VALUE_CST_COLUMNS
    ]


@dataclass
class MergeDiagnostics:
    """Summary statistics describing the quality of the master panel's merges."""

    n_rows: int
    n_unique_permno_months: int
    n_duplicate_permno_months: int
    pct_gvkey_matched: float
    pct_compustat_usable: float
    pct_compustat_expired: float
    pct_ff_matched: float
    n_delisted_rows: int
    n_delisted_missing_return: float
    accounting_var_missing_pct: dict[str, float] = field(default_factory=dict)
    firm_coverage_by_year: list[dict[str, object]] = field(default_factory=list)


def _pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 4) if denominator else 0.0


def compute_firm_coverage_by_year(panel: pd.DataFrame) -> list[dict[str, object]]:
    """Distinct-firm counts and match rates per calendar year — a survivorship-bias check."""
    df = panel.copy()
    df["_year"] = pd.to_datetime(df["MthCalDt"]).dt.year
    value_cols = _cst_value_columns(panel)
    has_compustat = (
        df[value_cols].notna().any(axis=1) if value_cols else pd.Series(False, index=df.index)
    )

    rows: list[dict[str, object]] = []
    for year, grp in df.groupby("_year"):
        with_gvkey = grp.loc[grp["gvkey"].notna(), "PERMNO"]
        with_compustat = grp.loc[has_compustat.loc[grp.index], "PERMNO"]
        rows.append(
            {
                "year": int(year),
                "n_permno": int(grp["PERMNO"].nunique()),
                "n_permno_with_gvkey": int(with_gvkey.nunique()),
                "n_permno_with_compustat": int(with_compustat.nunique()),
            }
        )
    return rows


def compute_diagnostics(panel: pd.DataFrame) -> MergeDiagnostics:
    """Compute the full set of merge diagnostics for a completed master panel."""
    n_rows = len(panel)
    key_cols = ["PERMNO", "MthCalDt"]
    n_unique = int(panel.drop_duplicates(subset=key_cols).shape[0])
    n_duplicate = n_rows - n_unique

    pct_gvkey_matched = _pct(int(panel["gvkey"].notna().sum()), n_rows)

    value_cols = _cst_value_columns(panel)
    has_compustat = (
        panel[value_cols].notna().any(axis=1) if value_cols else pd.Series(False, index=panel.index)
    )
    pct_compustat_usable = _pct(int(has_compustat.sum()), n_rows)
    pct_compustat_expired = _pct(int(panel["cst_expired"].sum()), n_rows)

    pct_ff_matched = _pct(int(panel["ff_mktrf"].notna().sum()), n_rows)

    n_delisted_rows = int(panel["is_delisted"].sum())
    n_delisted_missing_return = int(panel["delisting_return_missing"].sum())

    accounting_var_missing_pct = {
        col: _pct(int(panel[col].isna().sum()), n_rows) for col in value_cols
    }

    return MergeDiagnostics(
        n_rows=n_rows,
        n_unique_permno_months=n_unique,
        n_duplicate_permno_months=n_duplicate,
        pct_gvkey_matched=pct_gvkey_matched,
        pct_compustat_usable=pct_compustat_usable,
        pct_compustat_expired=pct_compustat_expired,
        pct_ff_matched=pct_ff_matched,
        n_delisted_rows=n_delisted_rows,
        n_delisted_missing_return=n_delisted_missing_return,
        accounting_var_missing_pct=accounting_var_missing_pct,
        firm_coverage_by_year=compute_firm_coverage_by_year(panel),
    )
