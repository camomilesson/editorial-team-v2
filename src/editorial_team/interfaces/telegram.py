"""Thin Telegram transport adapter for ConversationService."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

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
SUPPORTED_CHAT_TYPES = frozenset({ChatType.PRIVATE, ChatType.GROUP, ChatType.SUPERGROUP})
AsyncSleeper = Callable[[float], Awaitable[None]]


class HeartbeatStoreLifecycle(Protocol):
    """Synchronous initialization required by optional heartbeat startup."""

    def initialize(self) -> None: ...


class HeartbeatSchedulerLifecycle(Protocol):
    """Asynchronous lifecycle required by optional heartbeat startup."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...


def conversation_id_for_chat(
    chat_id: int,
    message_thread_id: int | None = None,
) -> str:
    """Return a stable conversation ID for one Telegram chat or forum topic."""

    if isinstance(chat_id, bool) or not isinstance(chat_id, int):
        raise ValueError("chat_id must be an integer")
    encoded = f"n{abs(chat_id)}" if chat_id < 0 else str(chat_id)
    conversation_id = f"telegram-chat-{encoded}"
    if message_thread_id is None:
        return conversation_id
    if isinstance(message_thread_id, bool) or not isinstance(message_thread_id, int):
        raise ValueError("message_thread_id must be an integer or None")
    thread = f"n{abs(message_thread_id)}" if message_thread_id < 0 else str(message_thread_id)
    return f"{conversation_id}-thread-{thread}"


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
        allowed_chat_ids: frozenset[int],
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
        if (
            not isinstance(allowed_chat_ids, frozenset)
            or not allowed_chat_ids
            or not all(
                isinstance(chat_id, int) and not isinstance(chat_id, bool) and chat_id != 0
                for chat_id in allowed_chat_ids
            )
        ):
            raise ValueError("allowed_chat_ids must be a non-empty frozenset of chat IDs")
        self._service = service
        self._runtime_queue = runtime_queue
        self._allowed_chat_ids = allowed_chat_ids
        self._handoff_delay = float(handoff_delay)
        self._sleeper = sleeper
        self._heartbeat_store: HeartbeatStoreLifecycle | None = None
        self._heartbeat_scheduler: HeartbeatSchedulerLifecycle | None = None

    def configure_heartbeat(
        self,
        *,
        store: HeartbeatStoreLifecycle,
        scheduler: HeartbeatSchedulerLifecycle,
    ) -> None:
        """Attach optional heartbeat lifecycle components before startup."""

        if self._heartbeat_store is not None or self._heartbeat_scheduler is not None:
            raise ValueError("Heartbeat lifecycle is already configured")
        self._heartbeat_store = store
        self._heartbeat_scheduler = scheduler

    async def start_runtime(self, application: Application) -> None:
        """Start the shared runtime worker with the Telegram application."""

        del application
        await self._runtime_queue.start()
        if self._heartbeat_store is not None and self._heartbeat_scheduler is not None:
            await asyncio.to_thread(self._heartbeat_store.initialize)
            await self._heartbeat_scheduler.start()

    async def close_runtime(self, application: Application) -> None:
        """Drain and close the shared runtime worker on application shutdown."""

        del application
        failure: BaseException | None = None
        try:
            if self._heartbeat_scheduler is not None:
                await self._heartbeat_scheduler.close()
        except BaseException as exc:
            failure = exc
        try:
            await self._runtime_queue.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
        close = getattr(self._service, "close", None)
        try:
            if callable(close):
                await asyncio.to_thread(close)
        except BaseException as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure

    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Send platform onboarding without invoking product behavior."""

        chat = update.effective_chat
        message = update.effective_message
        if not self._supported_chat(chat):
            return
        await context.bot.send_message(
            chat_id=chat.id,
            text=ONBOARDING_MESSAGE,
            **self._topic_kwargs(getattr(message, "message_thread_id", None)),
        )

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
            or not self._supported_chat(chat)
            or message is None
            or not isinstance(message.text, str)
        ):
            return

        trace = trace_for_update(update.update_id)

        async def operation() -> None:
            with bind_turn_trace(trace):
                await self._process_turn(
                    chat_id=chat.id,
                    message_thread_id=getattr(message, "message_thread_id", None),
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
        message_thread_id: int | None,
        text: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        trace_event("telegram_turn_started", stage="telegram")
        try:
            assistant_messages = await asyncio.to_thread(
                self._service.process_message,
                conversation_id_for_chat(
                    chat_id,
                    message_thread_id,
                ),
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
                message_thread_id=message_thread_id,
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
        message_thread_id: int | None,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> int:
        chunk_count = 0
        for index, message in enumerate(messages):
            if index:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    **self._topic_kwargs(message_thread_id),
                )
                await self._sleeper(self._handoff_delay)
            for chunk in chunk_text(message.content):
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    **self._topic_kwargs(message_thread_id),
                )
                chunk_count += 1
        return chunk_count

    def _supported_chat(self, chat: object) -> bool:
        chat_id = getattr(chat, "id", None)
        chat_type = getattr(chat, "type", None)
        return chat_type in SUPPORTED_CHAT_TYPES and chat_id in self._allowed_chat_ids

    @staticmethod
    def _topic_kwargs(message_thread_id: int | None) -> dict[str, int]:
        return {} if message_thread_id is None else {"message_thread_id": message_thread_id}


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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, adapter.handle_text))
    return application
