"""Registry describing the raw WRDS extracts this pipeline validates.

Each `DatasetSpec` records what was actually observed by inspecting the raw
files (column names, a working primary key, which columns are dates, and any
non-standard date sentinels) — it is documentation of the real files, not an
assumed schema. Column-level statistics (dtypes, null counts, etc.) are still
computed generically at runtime by `profiling.py` from whatever columns are
actually present in the CSV; nothing here hardcodes value-level assumptions.

Where a dataset has no reliable natural key (see `crsp_names`), `declared_key`
is left `None` and `profiling.suggest_primary_key` is used to search for one
automatically instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one raw dataset, as empirically determined from the file."""

    name: str
    filename: str
    date_columns: tuple[str, ...] = ()
    declared_key: tuple[str, ...] | None = None
    id_columns: tuple[str, ...] = ()
    date_sentinels: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "crsp_monthly_stock": DatasetSpec(
        name="crsp_monthly_stock",
        filename="crsp_monthly_stock.csv",
        date_columns=("MthCalDt",),
        declared_key=("PERMNO", "MthCalDt"),
        id_columns=("PERMNO", "PERMCO", "CUSIP", "HdrCUSIP", "Ticker"),
        notes=(
            "CRSP Monthly Stock File. One row per security (PERMNO) per "
            "month-end calendar date (MthCalDt)."
        ),
    ),
    "crsp_delisting": DatasetSpec(
        name="crsp_delisting",
        filename="crsp_delisting.csv",
        date_columns=("DelistingDt",),
        declared_key=("PERMNO",),
        id_columns=("PERMNO", "DelPERMNO"),
        notes=(
            "CRSP delisting events, one row per PERMNO in this extract. "
            "DelPERMNO == 0 is a sentinel meaning 'no successor security'."
        ),
    ),
    "crsp_names": DatasetSpec(
        name="crsp_names",
        filename="crsp_names.csv",
        date_columns=(),
        declared_key=None,
        id_columns=("permno", "permco", "cusip", "ticker"),
        notes=(
            "CRSP historical name/identifier records. This extract has no "
            "NAMEDT/NAMEENDDT validity-period columns, so individual rows "
            "cannot be date-bounded and there is no reliable row-level "
            "primary key — many rows are legitimately repeated verbatim."
        ),
    ),
    "ccm_link_table": DatasetSpec(
        name="ccm_link_table",
        filename="ccm_link_table.csv",
        date_columns=("LINKDT", "LINKENDDT"),
        declared_key=("gvkey", "LIID", "LINKDT"),
        id_columns=("gvkey", "LPERMNO", "LPERMCO"),
        date_sentinels={"LINKENDDT": {"E": "open-ended: link is still active"}},
        notes=(
            "CRSP-Compustat Merged (CCM) link table, mapping Compustat gvkey "
            "to CRSP permno/permco over a validity window [LINKDT, LINKENDDT]."
        ),
    ),
    "compustat_fundamentals_annual": DatasetSpec(
        name="compustat_fundamentals_annual",
        filename="compustat_fundamentals_annual.csv",
        date_columns=("datadate", "apdedate"),
        declared_key=("GVKEY", "datadate"),
        id_columns=("GVKEY",),
        notes=(
            "Compustat annual fundamentals (industrial format, consolidated, "
            "domestic, standardized, USD, in this extract)."
        ),
    ),
    "fama_french_5f_momentum_monthly": DatasetSpec(
        name="fama_french_5f_momentum_monthly",
        filename="fama_french_5f_momentum_monthly.csv",
        date_columns=("dateff",),
        declared_key=("dateff",),
        id_columns=(),
        notes="Fama-French 5 factors plus momentum (umd), monthly.",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a dataset spec by name, raising a clear error if unknown."""
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {sorted(DATASETS)}") from exc
