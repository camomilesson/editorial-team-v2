"""Thin Telegram transport adapter for ConversationService."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from editorial_team.conversation import ConversationService
from editorial_team.domain.conversation import Message
from editorial_team.runtime import (
    QueueCapacityError,
    RuntimeJobSource,
    RuntimeQueue,
)
from editorial_team.tracing import (
    bind_turn_trace,
    current_trace_stage,
    error_category,
    set_trace_stage,
    trace_event,
    trace_for_update,
)

MAX_TELEGRAM_TEXT_LENGTH = 4096
ONBOARDING_MESSAGE = (
    "Talker\n\nHi — welcome to Editorial Team. We can discuss, draft, review, "
    "revise, translate, or proofread text together."
)
GENERIC_TURN_ERROR = "Sorry — I couldn’t complete that turn. Please try again."
BUSY_TURN_ERROR = "Sorry — the team is busy right now. Please try again shortly."
DEFAULT_HANDOFF_DELAY_SECONDS = 1.25
AsyncSleeper = Callable[[float], Awaitable[None]]


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

    def __init__(
        self,
        service: ConversationService,
        runtime_queue: RuntimeQueue,
        *,
        handoff_delay: float = DEFAULT_HANDOFF_DELAY_SECONDS,
        sleeper: AsyncSleeper = asyncio.sleep,
    ) -> None:
        if (
            isinstance(handoff_delay, bool)
            or not isinstance(handoff_delay, (int, float))
            or not math.isfinite(handoff_delay)
            or handoff_delay < 0
        ):
            raise ValueError("handoff_delay must be a non-negative number")
        if not callable(sleeper):
            raise ValueError("sleeper must be callable")
        self._service = service
        self._runtime_queue = runtime_queue
        self._handoff_delay = float(handoff_delay)
        self._sleeper = sleeper

    async def start_runtime(self, application: Application) -> None:
        """Start the shared runtime worker with the Telegram application."""

        del application
        await self._runtime_queue.start()

    async def close_runtime(self, application: Application) -> None:
        """Drain and close the shared runtime worker on application shutdown."""

        del application
        await self._runtime_queue.close()

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

        trace = trace_for_update(update.update_id)
        async def operation() -> None:
            with bind_turn_trace(trace):
                await self._process_turn(
                    chat_id=chat.id,
                    text=message.text,
                    context=context,
                )

        try:
            await self._runtime_queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id=trace.correlation_id,
                operation=operation,
            )
        except QueueCapacityError:
            await context.bot.send_message(chat_id=chat.id, text=BUSY_TURN_ERROR)

    async def _process_turn(
        self,
        *,
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        trace_event("telegram_turn_started", stage="telegram")
        try:
            assistant_messages = await asyncio.to_thread(
                self._service.process_message,
                conversation_id_for_chat(chat_id),
                text,
            )
        except Exception as exc:
            trace_event(
                "telegram_turn_failed",
                stage=current_trace_stage(),
                outcome="failed",
                error_category=error_category(exc),
            )
            await context.bot.send_message(chat_id=chat_id, text=GENERIC_TURN_ERROR)
            return

        set_trace_stage("assistant_delivery")
        trace_event(
            "assistant_delivery_started",
            stage="assistant_delivery",
            assistant_message_count=len(assistant_messages),
        )
        try:
            chunk_count = await self._send_messages(
                chat_id=chat_id,
                messages=assistant_messages,
                context=context,
            )
        except Exception as exc:
            trace_event(
                "telegram_turn_failed",
                stage="assistant_delivery",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise
        trace_event(
            "assistant_delivery_completed",
            stage="assistant_delivery",
            outcome="completed",
            assistant_message_count=len(assistant_messages),
            chunk_count=chunk_count,
        )
        set_trace_stage("telegram")
        trace_event("telegram_turn_completed", stage="telegram", outcome="completed")

    async def _send_messages(
        self,
        *,
        chat_id: int,
        messages: Sequence[Message],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> int:
        chunk_count = 0
        for index, message in enumerate(messages):
            if index:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                )
                await self._sleeper(self._handoff_delay)
            for chunk in chunk_text(message.content):
                await context.bot.send_message(chat_id=chat_id, text=chunk)
                chunk_count += 1
        return chunk_count


def build_telegram_application(
    *,
    token: str,
    adapter: TelegramAdapter,
) -> Application:
    """Build a sequential long-polling Telegram application."""

    if not isinstance(token, str) or not token.strip():
        raise ValueError("Telegram token is required")
    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(False)
        .post_init(adapter.start_runtime)
        .post_shutdown(adapter.close_runtime)
        .build()
    )
    application.add_handler(CommandHandler("start", adapter.start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, adapter.handle_text)
    )
    return application
