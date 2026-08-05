"""Tests for explicit Telegram ingress configuration."""

import pytest

from editorial_team.app.telegram_config import (
    TelegramConfigurationError,
    load_telegram_configuration,
)


def test_allowlist_accepts_private_and_group_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_TELEGRAM_ALLOWED_CHAT_IDS", "123,-100,-200")
    assert load_telegram_configuration().allowed_chat_ids == frozenset({123, -100, -200})


@pytest.mark.parametrize("value", ["", " ", "abc", "0", "+123", "123, bad"])
def test_allowlist_rejects_missing_or_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EDITORIAL_TELEGRAM_ALLOWED_CHAT_IDS", value)
    with pytest.raises(TelegramConfigurationError, match="allowlist"):
        load_telegram_configuration()
