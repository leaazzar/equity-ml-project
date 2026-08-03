#!/usr/bin/env python
"""Entry point for the data ingestion/validation pipeline.

Usage:
    python scripts/run_data_validation.py
    python scripts/run_data_validation.py --raw-dir data/raw --reports-dir reports/data_validation

See src/data_validation/ for the implementation and tests/data_validation/
for the test suite.
"""

from __future__ import annotations

import sys

from data_validation.cli import main

if __name__ == "__main__":
    sys.exit(main())
