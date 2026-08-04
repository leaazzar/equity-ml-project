"""Loads the out-of-sample prediction panel (`equity_ml.models`' output) and
the realized-return / factor data the backtest scores it against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_ml.backtest.config import BacktestConfig
from equity_ml.models.config import ModelConfig
from equity_ml.models.io import load_master_panel, load_model_ready_features
from equity_ml.models.pipeline import build_training_panel


def load_predictions(modeling_reports_dir: Path | str) -> pd.DataFrame:
    """Load the out-of-sample prediction panel written by
    `equity_ml.models.reporting.write_reports`."""
    path = Path(modeling_reports_dir) / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}. Run scripts/run_modeling.py first."
        )
    return pd.read_parquet(path)


def load_realized_returns_and_factors(
    processed_dir: Path | str,
    model_config: ModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the raw (non-demeaned) forward-return target and load the
    Fama-French factor columns, both keyed by (permno, date).

    The training panel's `target_raw` — not `target_demeaned`, which is what
    models are trained on (MODEL_DESIGN.md Section 1) — is what the backtest
    needs: actual realized dollar return, not the market-neutralized
    training quantity. Rebuilt here (cheap — no model fitting involved)
    rather than persisted separately, since it's a deterministic function of
    already-written inputs.
    """
    model_config = model_config or ModelConfig()
    backtest_config = backtest_config or BacktestConfig()
    processed_dir = Path(processed_dir)

    master_panel = load_master_panel(processed_dir)
    model_ready_features = load_model_ready_features(processed_dir)
    panel = build_training_panel(master_panel, model_ready_features, model_config)

    realized_returns = panel[
        [model_config.permno_column, model_config.date_column, "target_raw"]
    ].rename(columns={"target_raw": backtest_config.realized_return_column})

    factor_columns = [c for c in backtest_config.factor_columns if c in master_panel.columns]
    factor_panel = master_panel[[model_config.date_column, *factor_columns]].drop_duplicates(
        subset=model_config.date_column
    )
    return realized_returns, factor_panel
