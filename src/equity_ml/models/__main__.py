"""Allows running the pipeline as `python -m equity_ml.models`."""

from __future__ import annotations

import sys

from equity_ml.models.cli import main

if __name__ == "__main__":
    sys.exit(main())
