"""Allows running the pipeline as `python -m equity_ml.backtest`."""

from __future__ import annotations

import sys

from equity_ml.backtest.cli import main

if __name__ == "__main__":
    sys.exit(main())
