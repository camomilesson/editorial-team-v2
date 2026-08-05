"""Sanitized Telegram ingress configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


class TelegramConfigurationError(RuntimeError):
    """Telegram ingress configuration is absent or invalid."""


@dataclass(frozen=True)
class TelegramConfiguration:
    """Explicit chat allowlist for product traffic."""

    allowed_chat_ids: frozenset[int]


def load_telegram_configuration() -> TelegramConfiguration:
    """Load a required comma-separated allowlist without echoing invalid values."""

    raw = os.getenv("EDITORIAL_TELEGRAM_ALLOWED_CHAT_IDS", "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise TelegramConfigurationError("Telegram chat allowlist is missing")
    try:
        chat_ids = frozenset(int(item) for item in values)
    except ValueError:
        raise TelegramConfigurationError("Telegram chat allowlist is invalid") from None
    if any(chat_id == 0 or str(chat_id) not in values for chat_id in chat_ids):
        raise TelegramConfigurationError("Telegram chat allowlist is invalid")
    return TelegramConfiguration(chat_ids)
