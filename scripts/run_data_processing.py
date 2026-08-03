#!/usr/bin/env python
"""Entry point for the point-in-time data integration pipeline.

Usage:
    python scripts/run_data_processing.py
    python scripts/run_data_processing.py --interim-dir data/interim --processed-dir data/processed

Requires data/interim/ to already exist — run scripts/run_data_validation.py
first if it doesn't. See src/data_processing/ for the implementation and
MERGE_REPORT.md for the methodology.
"""

from __future__ import annotations

import sys

from data_processing.cli import main

if __name__ == "__main__":
    sys.exit(main())
