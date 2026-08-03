"""Configuration loading utilities.

Provides project paths, a YAML config loader, and a thin `Settings` wrapper
around `configs/config.yaml`. No WRDS-specific schema logic lives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Load a local .env if present; never overrides variables already set in
# the environment (e.g. by CI or the shell).
load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Settings:
    """Top-level project settings, loaded from configs/config.yaml."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    reports_dir: Path = REPORTS_DIR
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path | str = CONFIG_DIR / "config.yaml") -> Settings:
        raw = load_yaml_config(config_path)
        return cls(raw=raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def wrds_username() -> str | None:
    """Return the WRDS username from the environment, if set.

    TODO(WRDS): Populate WRDS_USERNAME in a local .env file (never commit it)
    once WRDS access is provisioned. Authentication itself is handled by the
    `wrds` package via a .pgpass entry or interactive prompt.
    """
    return os.environ.get("WRDS_USERNAME")
