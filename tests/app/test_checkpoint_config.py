"""Tests for checkpoint path configuration."""

from pathlib import Path

import pytest

from editorial_team.app.checkpoint_config import (
    DEFAULT_CHECKPOINT_BUSY_TIMEOUT_SECONDS,
    DEFAULT_CHECKPOINT_DB_PATH,
    CheckpointConfigurationError,
    load_checkpoint_configuration,
)


def test_checkpoint_path_defaults_and_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITORIAL_CHECKPOINT_DB_PATH", raising=False)
    assert load_checkpoint_configuration().database_path == DEFAULT_CHECKPOINT_DB_PATH
    assert (
        load_checkpoint_configuration().busy_timeout_seconds
        == DEFAULT_CHECKPOINT_BUSY_TIMEOUT_SECONDS
    )
    monkeypatch.setenv("EDITORIAL_CHECKPOINT_DB_PATH", "local/conversations.db")
    assert load_checkpoint_configuration().database_path == Path("local/conversations.db")


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_busy_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EDITORIAL_CHECKPOINT_BUSY_TIMEOUT_SECONDS", value)
    with pytest.raises(CheckpointConfigurationError, match="busy timeout"):
        load_checkpoint_configuration()


def test_busy_timeout_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_CHECKPOINT_BUSY_TIMEOUT_SECONDS", "0.25")
    assert load_checkpoint_configuration().busy_timeout_seconds == 0.25


def test_blank_checkpoint_path_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_CHECKPOINT_DB_PATH", " ")
    with pytest.raises(CheckpointConfigurationError, match="Checkpoint database path is invalid"):
        load_checkpoint_configuration()
