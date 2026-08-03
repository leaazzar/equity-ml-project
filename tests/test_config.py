"""Tests for equity_ml.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from equity_ml.config import CONFIG_DIR, Settings, load_yaml_config, wrds_username


def test_load_yaml_config_reads_project_config() -> None:
    config = load_yaml_config(CONFIG_DIR / "config.yaml")
    assert config["project"]["name"] == "equity-ml"


def test_load_yaml_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "does_not_exist.yaml")


def test_settings_load_defaults() -> None:
    settings = Settings.load()
    assert settings.get("project", {})["random_seed"] == 42


def test_wrds_username_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WRDS_USERNAME", "test_user")
    assert wrds_username() == "test_user"


def test_wrds_username_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    assert wrds_username() is None
