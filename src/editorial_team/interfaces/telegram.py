"""Thin Telegram transport adapter for ConversationService."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from editorial_team.conversation import ConversationService
from editorial_team.domain.conversation import Message

MAX_TELEGRAM_TEXT_LENGTH = 4096
ONBOARDING_MESSAGE = (
    "Hi — just message me normally. I can chat with you or help draft and revise text."
)
GENERIC_TURN_ERROR = "Sorry — I couldn’t complete that turn. Please try again."

logger = logging.getLogger(__name__)


def conversation_id_for_chat(chat_id: int) -> str:
    """Convert a Telegram chat identifier into a stable opaque identifier."""

    if isinstance(chat_id, bool) or not isinstance(chat_id, int):
        raise ValueError("chat_id must be an integer")
    encoded = f"n{abs(chat_id)}" if chat_id < 0 else str(chat_id)
    return f"telegram-chat-{encoded}"


def chunk_text(text: str, limit: int = MAX_TELEGRAM_TEXT_LENGTH) -> tuple[str, ...]:
    """Split plain text without loss, preferring paragraph and line boundaries."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be nonblank")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = _preferred_split(remaining, limit)
        chunk = remaining[:split_at]
        if not chunk.strip():
            split_at = limit
            chunk = remaining[:split_at]
        if not chunk.strip():
            raise ValueError("text cannot be split into nonblank chunks")
        chunks.append(chunk)
        remaining = remaining[split_at:]

    if not remaining.strip():
        if not chunks or len(chunks[-1]) + len(remaining) > limit:
            raise ValueError("text cannot be split into nonblank chunks")
        chunks[-1] += remaining
    else:
        chunks.append(remaining)
    return tuple(chunks)


def _preferred_split(text: str, limit: int) -> int:
    paragraph = text.rfind("\n\n", 0, limit)
    if paragraph >= 0:
        return paragraph + 2
    line = text.rfind("\n", 0, limit)
    if line >= 0:
        return line + 1
    return limit


class TelegramAdapter:
    """Translate private Telegram text updates to conversation turns."""

    def __init__(self, service: ConversationService) -> None:
        self._service = service
        self._turn_lock = asyncio.Lock()

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send platform onboarding without invoking product behavior."""

        chat = update.effective_chat
        if chat is None or chat.type != ChatType.PRIVATE:
            return
        await context.bot.send_message(chat_id=chat.id, text=ONBOARDING_MESSAGE)

    async def handle_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Process one supported text update without blocking the event loop."""

        chat = update.effective_chat
        message = update.effective_message
        if (
            chat is None
            or chat.type != ChatType.PRIVATE
            or message is None
            or not isinstance(message.text, str)
        ):
            return

        async with self._turn_lock:
            try:
                assistant_messages = await asyncio.to_thread(
                    self._service.process_message,
                    conversation_id_for_chat(chat.id),
                    message.text,
                )
            except Exception as exc:
                logger.warning(
                    "telegram_turn_failed category=%s update_id=%s",
                    type(exc).__name__,
                    update.update_id,
                )
                await context.bot.send_message(chat_id=chat.id, text=GENERIC_TURN_ERROR)
                return

            await self._send_messages(
                chat_id=chat.id,
                messages=assistant_messages,
                context=context,
            )

    @staticmethod
    async def _send_messages(
        *,
        chat_id: int,
        messages: Sequence[Message],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        for message in messages:
            for chunk in chunk_text(message.content):
                await context.bot.send_message(chat_id=chat_id, text=chunk)


def build_telegram_application(
    *,
    token: str,
    adapter: TelegramAdapter,
) -> Application:
    """Build a sequential long-polling Telegram application."""

    if not isinstance(token, str) or not token.strip():
        raise ValueError("Telegram token is required")
    application = Application.builder().token(token).concurrent_updates(False).build()
    application.add_handler(CommandHandler("start", adapter.start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, adapter.handle_text)
    )
    return application
