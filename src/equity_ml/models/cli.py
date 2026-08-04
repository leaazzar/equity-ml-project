"""Command-line entry point for the Phase 4 modeling pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from equity_ml.config import DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from equity_ml.logging_utils import setup_logging
from equity_ml.models.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run purged/embargoed walk-forward model training and write "
        "out-of-sample predictions + diagnostics.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DATA_DIR / "processed",
        help="Directory containing master_panel.parquet and features_model_ready.parquet.",
    )
    parser.add_argument(
        "--reports-dir",
        default=REPORTS_DIR / "modeling",
        help="Directory to write predictions and diagnostics to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        result = run_pipeline(args.processed_dir, args.reports_dir)
    except FileNotFoundError:
        logger.exception(
            "Required input not found — run scripts/run_data_processing.py and "
            "scripts/run_feature_engineering.py first"
        )
        return 1

    n_failed = sum(1 for r in result.validation_results if not r.passed)
    if n_failed:
        logger.error("%d validation check(s) failed — see the diagnostics report", n_failed)

    n_total = len(result.validation_results)
    print(f"Predictions written to: {result.predictions_path.relative_to(PROJECT_ROOT)}")
    print(f"IC summary written to: {result.ic_summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Walk-forward folds: {len(result.folds)}")
    print(f"Validation: {n_total - n_failed}/{n_total} passed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
