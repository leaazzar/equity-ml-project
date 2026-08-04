#!/usr/bin/env python
"""Entry point for the Phase 4 walk-forward modeling pipeline.

Usage:
    python scripts/run_modeling.py

Requires data/processed/master_panel.parquet and
data/processed/features_model_ready.parquet (run
scripts/run_data_processing.py and scripts/run_feature_engineering.py
first). See src/equity_ml/models/ for the implementation and
MODEL_DESIGN.md for the full design.
"""

from __future__ import annotations

import sys

from equity_ml.models.cli import main

if __name__ == "__main__":
    sys.exit(main())
