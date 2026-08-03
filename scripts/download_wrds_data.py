#!/usr/bin/env python
"""Entry point for downloading WRDS data.

STATUS: NOT YET IMPLEMENTED. WRDS data has not been downloaded.

TODO(WRDS): Once WRDS access is provisioned and target datasets/schemas are
confirmed (see configs/data_sources.yaml and DATA_DICTIONARY.md), wire this
script up to equity_ml.data.wrds_loader to pull and persist raw extracts
under data/raw/ (which is gitignored).
"""

from __future__ import annotations

import argparse
import sys

from equity_ml.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WRDS data (TODO: not yet implemented).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="TODO(WRDS): dataset key from configs/data_sources.yaml",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    parse_args()
    logger.error(
        "WRDS download is not yet implemented. See TASKS.md and "
        "configs/data_sources.yaml for the current TODOs."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
