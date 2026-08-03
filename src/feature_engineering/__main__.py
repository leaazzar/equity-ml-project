"""Allows running the pipeline as `python -m feature_engineering`."""

from __future__ import annotations

import sys

from feature_engineering.cli import main

if __name__ == "__main__":
    sys.exit(main())
