"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    from equity_ml.config import PROJECT_ROOT

    return PROJECT_ROOT
