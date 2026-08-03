"""Command-line entry point for the point-in-time data integration pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from data_processing.pipeline import run_pipeline
from equity_ml.config import DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from equity_ml.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the point-in-time master panel from data/interim/."
    )
    parser.add_argument(
        "--interim-dir",
        default=DATA_DIR / "interim",
        help="Directory containing the typed/deduplicated interim Parquet files.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DATA_DIR / "processed",
        help="Directory to write the master panel Parquet file to.",
    )
    parser.add_argument(
        "--reports-dir",
        default=REPORTS_DIR / "data_processing",
        help="Directory to write merge diagnostics to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        result = run_pipeline(args.interim_dir, args.processed_dir, args.reports_dir)
    except FileNotFoundError:
        logger.exception("Interim data not found — run scripts/run_data_validation.py first")
        return 1

    logger.info(
        "Merge complete. Panel written to %s; diagnostics at %s",
        result.output_path,
        result.report_path,
    )
    print(f"Master panel written to: {result.output_path.relative_to(PROJECT_ROOT)}")
    print(f"Diagnostics written to: {result.report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
