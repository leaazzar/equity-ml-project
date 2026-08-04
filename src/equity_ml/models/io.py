"""Loads the master panel (source of `ret_adj`, needed for target
construction — not present in either feature panel) and the model-ready
feature panel this pipeline trains on.
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


def load_model_ready_features(processed_dir: Path | str) -> pd.DataFrame:
    """Load data/processed/features_model_ready.parquet (produced by feature_engineering)."""
    path = Path(processed_dir) / "features_model_ready.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Model-ready features not found: {path}. Run scripts/run_feature_engineering.py first."
        )
    return pd.read_parquet(path)
