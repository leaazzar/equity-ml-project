"""Allows running the pipeline as `python -m data_validation`."""

from __future__ import annotations

import sys

from data_validation.cli import main

if __name__ == "__main__":
    sys.exit(main())
