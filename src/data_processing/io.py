"""Loads the typed/deduplicated interim Parquet files this pipeline merges.

Intentionally reads from `data/interim/`, not `data/raw/`: the raw CSVs are
handled exclusively by `data_validation` (schema/date parsing, exact-duplicate
removal). This package assumes that pipeline has already been run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_DATASETS = (
    "crsp_monthly_stock",
    "crsp_delisting",
    "ccm_link_table",
    "compustat_fundamentals_annual",
    "fama_french_5f_momentum_monthly",
)


def load_interim(name: str, interim_dir: Path | str) -> pd.DataFrame:
    """Load one interim Parquet file by dataset name."""
    path = Path(interim_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Interim file not found for dataset {name!r}: {path}. "
            "Run `python -m data_validation` (or scripts/run_data_validation.py) "
            "first to generate data/interim/ from the raw WRDS extracts."
        )
    return pd.read_parquet(path)


def load_all_interim(interim_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load every dataset this pipeline needs, keyed by name.

    Note: `crsp_names` is intentionally excluded — it has no validity-period
    date columns (see DATA_DICTIONARY.md) and so cannot be joined point-in-time.
    """
    return {name: load_interim(name, interim_dir) for name in REQUIRED_DATASETS}
