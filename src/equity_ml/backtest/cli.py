"""Command-line entry point for the Phase 4 backtest pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from equity_ml.backtest.pipeline import run_pipeline
from equity_ml.config import DATA_DIR, PROJECT_ROOT, REPORTS_DIR
from equity_ml.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build long/short portfolios from out-of-sample predictions, backtest "
        "them, and write performance diagnostics.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DATA_DIR / "processed",
        help="Directory containing master_panel.parquet and features_model_ready.parquet.",
    )
    parser.add_argument(
        "--modeling-reports-dir",
        default=REPORTS_DIR / "modeling",
        help="Directory containing predictions.parquet (written by scripts/run_modeling.py).",
    )
    parser.add_argument(
        "--reports-dir",
        default=REPORTS_DIR / "backtest",
        help="Directory to write backtest diagnostics to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        result = run_pipeline(args.processed_dir, args.modeling_reports_dir, args.reports_dir)
    except FileNotFoundError:
        logger.exception("Required input not found — run scripts/run_modeling.py first")
        return 1

    print(f"Backtest summary written to: {result.summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
