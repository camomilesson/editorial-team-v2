"""Tests for sanitized optional heartbeat environment configuration."""

from pathlib import Path

import pytest

from editorial_team.app import (
    DEFAULT_HEARTBEAT_DB_PATH,
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    HeartbeatConfigurationError,
    load_heartbeat_configuration,
)

VARIABLES = (
    "EDITORIAL_HEARTBEAT_ENABLED",
    "EDITORIAL_HEARTBEAT_INTERVAL_SECONDS",
    "EDITORIAL_HEARTBEAT_DB_PATH",
    "EDITORIAL_ADMIN_TELEGRAM_CHAT_ID",
)


def clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_disabled_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch)

    config = load_heartbeat_configuration()

    assert config.enabled is False
    assert config.interval_seconds == DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    assert config.database_path == DEFAULT_HEARTBEAT_DB_PATH
    assert config.maintainer_chat_id is None


def test_valid_enabled_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    clear(monkeypatch)
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "true")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_DB_PATH", "custom/heartbeat.db")
    monkeypatch.setenv("EDITORIAL_ADMIN_TELEGRAM_CHAT_ID", "-100123")

    config = load_heartbeat_configuration()

    assert config.enabled is True
    assert config.interval_seconds == 60
    assert config.database_path == Path("custom/heartbeat.db")
    assert config.maintainer_chat_id == -100123


def test_enabled_without_maintainer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear(monkeypatch)
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "true")

    with pytest.raises(HeartbeatConfigurationError) as error:
        load_heartbeat_configuration()

    assert str(error.value) == "Enabled heartbeat requires maintainer configuration"


@pytest.mark.parametrize("value", ["abc", "0", "1.5", "+123"])
def test_invalid_chat_id_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear(monkeypatch)
    monkeypatch.setenv("EDITORIAL_ADMIN_TELEGRAM_CHAT_ID", value)

    with pytest.raises(HeartbeatConfigurationError) as error:
        load_heartbeat_configuration()

    assert value not in str(error.value)


@pytest.mark.parametrize("value", ["0", "-1", "9.9", "nan", "inf", "not-number"])
def test_invalid_live_interval_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear(monkeypatch)
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_INTERVAL_SECONDS", value)

    with pytest.raises(HeartbeatConfigurationError) as error:
        load_heartbeat_configuration()

    assert str(error.value) == "Heartbeat interval is invalid"
    assert value not in str(error.value)


def test_blank_database_path_is_rejected_without_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear(monkeypatch)
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_DB_PATH", " ")

    with pytest.raises(HeartbeatConfigurationError) as error:
        load_heartbeat_configuration()

    assert str(error.value) == "Heartbeat database path is invalid"
