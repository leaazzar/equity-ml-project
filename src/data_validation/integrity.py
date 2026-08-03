"""Cross-dataset referential integrity and consistency checks.

Unlike `profiling.py`, these checks are dataset-specific: they know that,
say, `ccm_link_table.LPERMNO` should be found in `crsp_monthly_stock.PERMNO`.
That mapping was determined by inspecting the actual raw files (see
`DATA_DICTIONARY.md`), not assumed from generic WRDS documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data_validation.datasets import DATASETS
from data_validation.io import parse_date_column

OPEN_ENDED_SENTINEL_DATE = pd.Timestamp("2099-12-31")
MAX_SAMPLES = 10


@dataclass
class IntegrityFinding:
    """Result of one cross-dataset (or single-dataset business-rule) check."""

    check_name: str
    datasets: list[str]
    status: str  # "ok", "info", or "warning"
    n_affected: int
    n_total: int
    description: str
    sample: list[str] = field(default_factory=list)


def _finding(
    check_name: str,
    datasets: list[str],
    n_affected: int,
    n_total: int,
    description: str,
    sample: list[str] | None = None,
    warn_if_any: bool = True,
) -> IntegrityFinding:
    status = "ok" if n_affected == 0 else ("warning" if warn_if_any else "info")
    return IntegrityFinding(
        check_name=check_name,
        datasets=datasets,
        status=status,
        n_affected=n_affected,
        n_total=n_total,
        description=description,
        sample=sample or [],
    )


def _ccm_linkenddt_parsed(ccm: pd.DataFrame) -> pd.Series:
    spec = DATASETS["ccm_link_table"]
    result = parse_date_column(ccm["LINKENDDT"], "LINKENDDT", spec)
    parsed = result.parsed.copy()
    is_sentinel = ccm["LINKENDDT"].isin(spec.date_sentinels["LINKENDDT"].keys())
    parsed.loc[is_sentinel] = OPEN_ENDED_SENTINEL_DATE
    return parsed


def check_delisting_permno_in_crsp(crsp: pd.DataFrame, delisting: pd.DataFrame) -> IntegrityFinding:
    """Every crsp_delisting.PERMNO should reference a security present in CRSP monthly."""
    crsp_permnos = set(crsp["PERMNO"].unique())
    missing = delisting.loc[~delisting["PERMNO"].isin(crsp_permnos), "PERMNO"]
    return _finding(
        check_name="delisting_permno_in_crsp_monthly",
        datasets=["crsp_delisting", "crsp_monthly_stock"],
        n_affected=len(missing),
        n_total=len(delisting),
        description=("crsp_delisting rows whose PERMNO does not appear in crsp_monthly_stock."),
        sample=[str(v) for v in missing.unique()[:MAX_SAMPLES]],
    )


def check_delisting_missing_returns(delisting: pd.DataFrame) -> IntegrityFinding:
    """Delisting records with a null delisting return (DelRet) — missing delisting info."""
    missing = delisting["DelRet"].isna()
    return _finding(
        check_name="delisting_missing_delret",
        datasets=["crsp_delisting"],
        n_affected=int(missing.sum()),
        n_total=len(delisting),
        description="crsp_delisting rows with a null DelRet (delisting return unavailable).",
        sample=[str(v) for v in delisting.loc[missing, "PERMNO"].unique()[:MAX_SAMPLES]],
    )


def check_delisting_successor_permno_coverage(
    crsp: pd.DataFrame, delisting: pd.DataFrame
) -> IntegrityFinding:
    """Non-zero DelPERMNO (successor security) values should also exist in CRSP monthly."""
    crsp_permnos = set(crsp["PERMNO"].unique())
    nonzero = delisting.loc[delisting["DelPERMNO"] != 0, "DelPERMNO"]
    missing = nonzero.loc[~nonzero.isin(crsp_permnos)]
    return _finding(
        check_name="delisting_successor_permno_in_crsp_monthly",
        datasets=["crsp_delisting", "crsp_monthly_stock"],
        n_affected=len(missing),
        n_total=len(nonzero),
        description=(
            "crsp_delisting rows with a non-zero DelPERMNO (successor security) "
            "that does not appear in crsp_monthly_stock."
        ),
        sample=[str(v) for v in missing.unique()[:MAX_SAMPLES]],
    )


def check_possible_missing_delisting_records(
    crsp: pd.DataFrame, delisting: pd.DataFrame, months_buffer: int = 6
) -> IntegrityFinding:
    """Heuristic: securities whose last CRSP observation is well before the sample
    end date, but with no corresponding delisting record, may be missing delisting
    information (or may simply still be halted/inactive without a formal delisting
    event in this extract).
    """
    dates = pd.to_datetime(crsp["MthCalDt"])
    last_obs = dates.groupby(crsp["PERMNO"]).max()
    cutoff = dates.max() - pd.DateOffset(months=months_buffer)
    candidates = last_obs[last_obs < cutoff]
    delisted_permnos = set(delisting["PERMNO"].unique())
    missing = candidates.index.difference(pd.Index(list(delisted_permnos)))
    return _finding(
        check_name="possible_missing_delisting_records",
        datasets=["crsp_monthly_stock", "crsp_delisting"],
        n_affected=len(missing),
        n_total=len(candidates),
        description=(
            f"PERMNOs with no CRSP observation in the last {months_buffer} months of "
            "the sample and no matching crsp_delisting record (heuristic candidates "
            "for missing delisting information)."
        ),
        sample=[str(v) for v in list(missing)[:MAX_SAMPLES]],
    )


def check_ccm_permno_coverage(ccm: pd.DataFrame, crsp: pd.DataFrame) -> IntegrityFinding:
    """CCM LPERMNO values should generally be found in crsp_monthly_stock.

    Mismatches whose entire link window ends before crsp_monthly_stock's earliest
    date are expected (CCM's history predates this CRSP extract's start) and are
    reported separately from the residual, unexplained gap.
    """
    crsp_permnos = set(crsp["PERMNO"].unique())
    crsp_min_date = pd.to_datetime(crsp["MthCalDt"]).min()
    linkenddt = _ccm_linkenddt_parsed(ccm)

    missing_mask = ~ccm["LPERMNO"].isin(crsp_permnos)
    unique_missing = ccm.loc[missing_mask, "LPERMNO"].unique()
    max_end_by_permno = linkenddt.groupby(ccm["LPERMNO"]).max()
    explained = (max_end_by_permno.loc[unique_missing] < crsp_min_date).sum()
    unexplained = len(unique_missing) - int(explained)

    return _finding(
        check_name="ccm_lpermno_in_crsp_monthly",
        datasets=["ccm_link_table", "crsp_monthly_stock"],
        n_affected=unexplained,
        n_total=len(unique_missing),
        description=(
            f"Unique CCM LPERMNO values not found in crsp_monthly_stock: "
            f"{len(unique_missing)} total, of which {explained} have their entire "
            f"link window ending before crsp_monthly_stock's earliest date "
            f"({crsp_min_date.date()}) and are therefore expected coverage gaps. "
            f"{unexplained} are unexplained by that gap alone."
        ),
    )


def check_ccm_gvkey_coverage(ccm: pd.DataFrame, compustat: pd.DataFrame) -> IntegrityFinding:
    """CCM gvkey values should generally be found in compustat_fundamentals_annual.

    As with permno coverage, mismatches explained by CCM's link window ending
    before Compustat's earliest datadate are reported separately.
    """
    cs_gvkeys = set(compustat["GVKEY"].unique())
    cs_min_date = pd.to_datetime(compustat["datadate"]).min()
    linkenddt = _ccm_linkenddt_parsed(ccm)

    missing_mask = ~ccm["gvkey"].isin(cs_gvkeys)
    unique_missing = ccm.loc[missing_mask, "gvkey"].unique()
    max_end_by_gvkey = linkenddt.groupby(ccm["gvkey"]).max()
    explained = (max_end_by_gvkey.loc[unique_missing] < cs_min_date).sum()
    unexplained = len(unique_missing) - int(explained)

    return _finding(
        check_name="ccm_gvkey_in_compustat",
        datasets=["ccm_link_table", "compustat_fundamentals_annual"],
        n_affected=unexplained,
        n_total=len(unique_missing),
        description=(
            f"Unique CCM gvkey values not found in compustat_fundamentals_annual: "
            f"{len(unique_missing)} total, of which {explained} have their entire "
            f"link window ending before Compustat's earliest datadate "
            f"({cs_min_date.date()}) and are therefore expected coverage gaps. "
            f"{unexplained} are unexplained by that gap alone."
        ),
    )


def check_compustat_gvkey_in_ccm(ccm: pd.DataFrame, compustat: pd.DataFrame) -> IntegrityFinding:
    """Every Compustat gvkey should have at least one CCM link row (needed to reach CRSP)."""
    ccm_gvkeys = set(ccm["gvkey"].unique())
    cs_gvkeys = compustat["GVKEY"].unique()
    missing = [g for g in cs_gvkeys if g not in ccm_gvkeys]
    return _finding(
        check_name="compustat_gvkey_in_ccm",
        datasets=["compustat_fundamentals_annual", "ccm_link_table"],
        n_affected=len(missing),
        n_total=len(cs_gvkeys),
        description=(
            "Unique Compustat GVKEY values with no row at all in ccm_link_table "
            "(these firm-years cannot be linked to a CRSP permno)."
        ),
        sample=[str(v) for v in missing[:MAX_SAMPLES]],
    )


def check_ccm_link_period_overlaps(ccm: pd.DataFrame) -> IntegrityFinding:
    """Within a given (gvkey, LIID), link validity windows should not overlap.

    Overlapping windows would mean two simultaneously "valid" links for the same
    security, which is ambiguous for merging CRSP and Compustat data.
    """
    linkdt = pd.to_datetime(ccm["LINKDT"])
    linkenddt = _ccm_linkenddt_parsed(ccm)
    frame = pd.DataFrame(
        {"gvkey": ccm["gvkey"], "LIID": ccm["LIID"], "start": linkdt, "end": linkenddt}
    )

    overlap_groups: list[str] = []
    for key, grp in frame.groupby(["gvkey", "LIID"]):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("start")
        ends = grp_sorted["end"].to_numpy()
        starts = grp_sorted["start"].to_numpy()
        if (ends[:-1] > starts[1:]).any():
            overlap_groups.append(f"gvkey={key[0]}, LIID={key[1]}")

    return _finding(
        check_name="ccm_link_period_overlaps",
        datasets=["ccm_link_table"],
        n_affected=len(overlap_groups),
        n_total=frame.groupby(["gvkey", "LIID"]).ngroups,
        description=(
            "(gvkey, LIID) groups in ccm_link_table with overlapping "
            "[LINKDT, LINKENDDT] validity windows."
        ),
        sample=overlap_groups[:MAX_SAMPLES],
    )


def check_fama_french_date_coverage(ff: pd.DataFrame, crsp: pd.DataFrame) -> IntegrityFinding:
    """CRSP monthly month-ends and Fama-French dateff months should match 1:1."""
    crsp_months = set(pd.to_datetime(crsp["MthCalDt"]).dt.to_period("M"))
    ff_months = set(pd.to_datetime(ff["dateff"]).dt.to_period("M"))
    missing_from_ff = sorted(crsp_months - ff_months)
    return _finding(
        check_name="fama_french_month_coverage",
        datasets=["fama_french_5f_momentum_monthly", "crsp_monthly_stock"],
        n_affected=len(missing_from_ff),
        n_total=len(crsp_months),
        description=(
            "Calendar months present in crsp_monthly_stock but missing from the Fama-French file."
        ),
        sample=[str(m) for m in missing_from_ff[:MAX_SAMPLES]],
    )


def check_identifier_naming_consistency() -> IntegrityFinding:
    """Document that the same real-world identifier is spelled differently across files.

    This is a static, structural observation (not data-dependent): PERMNO appears
    as `PERMNO` (crsp_monthly_stock, crsp_delisting), `permno` (crsp_names), and
    `LPERMNO` (ccm_link_table); gvkey appears as `GVKEY` (compustat) and `gvkey`
    (ccm_link_table). Downstream joins must not assume identical column names.
    """
    variants = {
        "permno": ["PERMNO", "permno", "LPERMNO"],
        "permco": ["PERMCO", "permco", "LPERMCO"],
        "gvkey": ["GVKEY", "gvkey"],
    }
    inconsistent = {k: v for k, v in variants.items() if len(set(v)) > 1}
    return _finding(
        check_name="identifier_naming_consistency",
        datasets=list(DATASETS.keys()),
        n_affected=len(inconsistent),
        n_total=len(variants),
        description=(
            "Identifiers whose column name casing/spelling differs across datasets: "
            f"{inconsistent}. Joins must map these explicitly rather than assuming a "
            "shared column name."
        ),
        warn_if_any=False,
    )


def run_all_integrity_checks(frames: dict[str, pd.DataFrame]) -> list[IntegrityFinding]:
    """Run every cross-dataset check against the loaded raw frames.

    `frames` must contain keys: crsp_monthly_stock, crsp_delisting, crsp_names,
    ccm_link_table, compustat_fundamentals_annual, fama_french_5f_momentum_monthly.
    """
    crsp = frames["crsp_monthly_stock"]
    delisting = frames["crsp_delisting"]
    ccm = frames["ccm_link_table"]
    compustat = frames["compustat_fundamentals_annual"]
    ff = frames["fama_french_5f_momentum_monthly"]

    return [
        check_delisting_permno_in_crsp(crsp, delisting),
        check_delisting_missing_returns(delisting),
        check_delisting_successor_permno_coverage(crsp, delisting),
        check_possible_missing_delisting_records(crsp, delisting),
        check_ccm_permno_coverage(ccm, crsp),
        check_ccm_gvkey_coverage(ccm, compustat),
        check_compustat_gvkey_in_ccm(ccm, compustat),
        check_ccm_link_period_overlaps(ccm),
        check_fama_french_date_coverage(ff, crsp),
        check_identifier_naming_consistency(),
    ]
