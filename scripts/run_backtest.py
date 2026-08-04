#!/usr/bin/env python
"""Entry point for the Phase 4 backtest pipeline.

Usage:
    python scripts/run_backtest.py

Requires reports/modeling/predictions.parquet (run
scripts/run_modeling.py first). See src/equity_ml/backtest/ for the
implementation and MODEL_DESIGN.md for the full design.
"""

from __future__ import annotations

import sys

from equity_ml.backtest.cli import main

if __name__ == "__main__":
    sys.exit(main())
