"""Tests for equity_ml.logging_utils."""

from __future__ import annotations

import logging
from pathlib import Path

from equity_ml.logging_utils import get_logger, setup_logging


def test_setup_logging_configures_root_logger() -> None:
    setup_logging()
    logger = get_logger("equity_ml.test")
    assert isinstance(logger, logging.Logger)


def test_setup_logging_falls_back_when_config_missing(tmp_path: Path) -> None:
    missing_config = tmp_path / "does_not_exist.yaml"
    setup_logging(config_path=missing_config)
    logger = get_logger("equity_ml.test_fallback")
    assert isinstance(logger, logging.Logger)
