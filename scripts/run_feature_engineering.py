#!/usr/bin/env python
"""Entry point for the point-in-time feature engineering pipeline.

Usage:
    python scripts/run_feature_engineering.py

Requires data/processed/master_panel.parquet (run
scripts/run_data_processing.py first). See src/feature_engineering/ for the
implementation and FEATURE_DICTIONARY.md for what each feature means.
"""

from __future__ import annotations

import sys

from feature_engineering.cli import main

if __name__ == "__main__":
    sys.exit(main())
