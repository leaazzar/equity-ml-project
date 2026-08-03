"""Loads the master panel (and, for growth-rate construction, the raw
Compustat interim table) this pipeline builds features from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_master_panel(processed_dir: Path | str) -> pd.DataFrame:
    """Load data/processed/master_panel.parquet (produced by data_processing)."""
    path = Path(processed_dir) / "master_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Master panel not found: {path}. Run scripts/run_data_processing.py first."
        )
    return pd.read_parquet(path)


def load_compustat_interim(interim_dir: Path | str) -> pd.DataFrame:
    """Load the raw (point-in-time-unmerged) Compustat annual fundamentals table.

    Needed to compute year-over-year growth rates on the actual gvkey-fiscal-year
    sequence (see accounting.py) before those growth rates are merged into the
    monthly panel using the same point-in-time logic as every other Compustat field.
    """
    path = Path(interim_dir) / "compustat_fundamentals_annual.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Interim Compustat file not found: {path}. Run scripts/run_data_validation.py first."
        )
    return pd.read_parquet(path)
