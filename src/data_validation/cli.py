"""Command-line entry point for the data ingestion/validation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from data_validation.pipeline import run_pipeline
from equity_ml.config import DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from equity_ml.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw WRDS extracts and write typed interim copies."
    )
    parser.add_argument(
        "--raw-dir", default=DATA_DIR / "raw", help="Directory containing the raw CSV files."
    )
    parser.add_argument(
        "--interim-dir",
        default=DATA_DIR / "interim",
        help="Directory to write cleaned/typed Parquet copies to.",
    )
    parser.add_argument(
        "--reports-dir",
        default=REPORTS_DIR / "data_validation",
        help="Directory to write validation reports to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        result = run_pipeline(args.raw_dir, args.interim_dir, args.reports_dir)
    except FileNotFoundError:
        logger.exception("Raw data not found — see PROJECT_ROOT/data/raw/README.md")
        return 1

    n_warnings = sum(1 for f in result.findings if f.status == "warning")
    logger.info(
        "Validation complete. %d dataset(s) profiled, %d integrity warning(s), report at %s",
        len(result.profiles),
        n_warnings,
        result.report_path,
    )
    print(f"Report written to: {result.report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
