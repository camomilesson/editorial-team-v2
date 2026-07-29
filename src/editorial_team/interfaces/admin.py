"""Telegram delivery adapter for deterministic maintainer notifications."""

from __future__ import annotations

from typing import Protocol

from editorial_team.errors import ServiceError
from editorial_team.interfaces.telegram import chunk_text
from editorial_team.operations.models import HeartbeatResult
from editorial_team.operations.notification import render_admin_notification


class TelegramBot(Protocol):
    """The narrow Telegram capability used for maintainer delivery."""

    async def send_message(self, *, chat_id: int, text: str) -> object: ...


class MaintainerNotificationError(ServiceError):
    """A sanitized maintainer Telegram delivery failure."""


class TelegramMaintainerNotifier:
    """Send deterministic operational alerts to one injected maintainer chat."""

    def __init__(self, bot: TelegramBot, maintainer_chat_id: int) -> None:
        if (
            isinstance(maintainer_chat_id, bool)
            or not isinstance(maintainer_chat_id, int)
            or maintainer_chat_id == 0
        ):
            raise ValueError("maintainer_chat_id must be a nonzero integer")
        self._bot = bot
        self._maintainer_chat_id = maintainer_chat_id

    async def notify(self, result: HeartbeatResult) -> None:
        """Send exactly one deterministic alert, chunked only if required."""

        text = render_admin_notification(result)
        try:
            for chunk in chunk_text(text):
                await self._bot.send_message(
                    chat_id=self._maintainer_chat_id,
                    text=chunk,
                )
        except Exception:
            raise MaintainerNotificationError(
                "Maintainer notification delivery failed"
            ) from None
