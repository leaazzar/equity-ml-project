"""Project-wide logging setup, driven by configs/logging.yaml."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from equity_ml.config import CONFIG_DIR, PROJECT_ROOT

DEFAULT_LOGGING_CONFIG = CONFIG_DIR / "logging.yaml"


def setup_logging(config_path: Path | str = DEFAULT_LOGGING_CONFIG) -> None:
    """Configure logging for the whole project from a YAML dictConfig file.

    Falls back to `logging.basicConfig` if the config file is missing, so
    imports never fail solely due to a missing/misconfigured logging file.
    """
    config_path = Path(config_path)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    if not config_path.exists():
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(
            "Logging config not found at %s; falling back to basicConfig.", config_path
        )
        return

    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call `setup_logging()` once at entry points."""
    return logging.getLogger(name)
