from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from telegram.constants import ChatAction, ChatType

from editorial_team.domain.conversation import Message, MessageRole
from editorial_team.interfaces.telegram import (
    BUSY_TURN_ERROR,
    DEFAULT_HANDOFF_DELAY_SECONDS,
    GENERIC_TURN_ERROR,
    MAX_TELEGRAM_TEXT_LENGTH,
    ONBOARDING_MESSAGE,
    TelegramAdapter,
    chunk_text,
    conversation_id_for_chat,
)
from editorial_team.runtime import QueueCapacityError, RuntimeJobSource, RuntimeQueue

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
ALLOWED = frozenset({123, -100, -200})


def assistant(content: str, number: int = 1) -> Message:
    return Message(
        f"message-{number}",
        "telegram-chat-123",
        MessageRole.ASSISTANT,
        content,
        NOW,
    )


@dataclass
class FakeChat:
    id: int = 123
    type: str = ChatType.PRIVATE


@dataclass
class FakeTelegramMessage:
    text: object
    message_thread_id: int | None = None


@dataclass
class FakeUpdate:
    update_id: int = 99
    effective_chat: FakeChat | None = None
    effective_message: FakeTelegramMessage | None = None


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.actions: list[dict[str, object]] = []
        self.events: list[tuple[str, object]] = []

    async def send_message(self, **kwargs: object) -> None:
        self.sent.append(kwargs)
        self.events.append(("message", kwargs["text"]))

    async def send_chat_action(self, **kwargs: object) -> None:
        self.actions.append(kwargs)
        self.events.append(("action", kwargs["action"]))


class RecordingSleeper:
    def __init__(self, events: list[tuple[str, object]] | None = None) -> None:
        self.calls: list[float] = []
        self.events = events

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)
        if self.events is not None:
            self.events.append(("sleep", delay))


class RecordingService:
    def __init__(self, output: tuple[Message, ...] = ()) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
        self.calls.append((conversation_id, text))
        return self.output


class ImmediateRuntimeQueue:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []
        self.starts = 0
        self.closes = 0

    async def start(self) -> None:
        self.starts += 1

    async def close(self) -> None:
        self.closes += 1

    async def submit(self, **kwargs: object) -> object:
        self.submissions.append(kwargs)
        operation = kwargs["operation"]
        return await operation()  # type: ignore[operator]


class RejectingRuntimeQueue(ImmediateRuntimeQueue):
    async def submit(self, **kwargs: object) -> object:
        self.submissions.append(kwargs)
        raise QueueCapacityError("private queue detail")


def context() -> SimpleNamespace:
    return SimpleNamespace(bot=FakeBot())


def private_update(text: object = "Exact user text", *, chat_id: int = 123) -> FakeUpdate:
    return FakeUpdate(
        effective_chat=FakeChat(chat_id),
        effective_message=FakeTelegramMessage(text),
    )


def test_conversation_identifier_is_stable_and_opaque() -> None:
    assert conversation_id_for_chat(123) == "telegram-chat-123"
    assert conversation_id_for_chat(123) == conversation_id_for_chat(123)
    assert conversation_id_for_chat(-456) == "telegram-chat-n456"
    assert conversation_id_for_chat(-456, 7) == "telegram-chat-n456-thread-7"
    assert conversation_id_for_chat(-456, 8) != conversation_id_for_chat(-456, 7)

    with pytest.raises(ValueError):
        conversation_id_for_chat(True)
    with pytest.raises(ValueError):
        conversation_id_for_chat(123, True)


@pytest.mark.parametrize("delay", [-1, math.nan, math.inf, True, "slow"])
def test_handoff_delay_must_be_a_non_negative_finite_number(delay: object) -> None:
    with pytest.raises(ValueError, match="handoff_delay"):
        TelegramAdapter(
            RecordingService(),
            ImmediateRuntimeQueue(),  # type: ignore[arg-type]
            allowed_chat_ids=ALLOWED,
            handoff_delay=delay,  # type: ignore[arg-type]
        )


def test_three_agent_messages_are_staged_in_order() -> None:
    service = RecordingService(
        (
            assistant("Writer output", 1),
            assistant("Critic evaluation", 2),
            assistant("Editor output", 3),
        )
    )
    ctx = context()
    sleeper = RecordingSleeper(ctx.bot.events)
    runtime_queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service,
        runtime_queue,
        allowed_chat_ids=ALLOWED,
        handoff_delay=0.25,
        sleeper=sleeper,
    )
    update = private_update()

    asyncio.run(adapter.handle_text(update, ctx))  # type: ignore[arg-type]

    assert service.calls == [("telegram-chat-123", "Exact user text")]
    assert [item["text"] for item in ctx.bot.sent] == [
        "Writer output",
        "Critic evaluation",
        "Editor output",
    ]
    assert all(item["chat_id"] == 123 for item in ctx.bot.sent)
    assert ctx.bot.events == [
        ("message", "Writer output"),
        ("action", ChatAction.TYPING),
        ("sleep", 0.25),
        ("message", "Critic evaluation"),
        ("action", ChatAction.TYPING),
        ("sleep", 0.25),
        ("message", "Editor output"),
    ]
    assert sleeper.calls == [0.25, 0.25]
    assert len(runtime_queue.submissions) == 1
    assert runtime_queue.submissions[0]["source"] is RuntimeJobSource.TELEGRAM
    assert runtime_queue.submissions[0]["correlation_id"] == "tg-99"


def test_one_talker_message_has_no_handoff_delay() -> None:
    service = RecordingService((assistant("Talker response"),))
    ctx = context()
    sleeper = RecordingSleeper(ctx.bot.events)
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service,
        ImmediateRuntimeQueue(),
        allowed_chat_ids=ALLOWED,
        sleeper=sleeper,
    )

    asyncio.run(adapter.handle_text(private_update(), ctx))  # type: ignore[arg-type]

    assert [item["text"] for item in ctx.bot.sent] == ["Talker response"]
    assert ctx.bot.actions == []
    assert sleeper.calls == []


def test_queue_capacity_rejection_sends_only_generic_busy_response() -> None:
    service = RecordingService((assistant("must not run"),))
    runtime_queue = RejectingRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service, runtime_queue, allowed_chat_ids=ALLOWED
    )
    ctx = context()

    asyncio.run(adapter.handle_text(private_update("PRIVATE USER TEXT"), ctx))  # type: ignore[arg-type]

    assert service.calls == []
    assert ctx.bot.sent == [{"chat_id": 123, "text": BUSY_TURN_ERROR}]
    assert len(runtime_queue.submissions) == 1


def test_adapter_lifecycle_starts_and_closes_injected_queue() -> None:
    runtime_queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        RecordingService(), runtime_queue, allowed_chat_ids=ALLOWED
    )
    application = SimpleNamespace()

    asyncio.run(adapter.start_runtime(application))  # type: ignore[arg-type]
    asyncio.run(adapter.close_runtime(application))  # type: ignore[arg-type]

    assert runtime_queue.starts == runtime_queue.closes == 1


def test_chunks_preserve_order_across_application_messages() -> None:
    first = "A" * (MAX_TELEGRAM_TEXT_LENGTH + 2)
    service = RecordingService((assistant(first, 1), assistant("Second", 2)))
    ctx = context()
    sleeper = RecordingSleeper(ctx.bot.events)
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service,
        ImmediateRuntimeQueue(),
        allowed_chat_ids=ALLOWED,
        sleeper=sleeper,
    )

    asyncio.run(adapter.handle_text(private_update(), ctx))  # type: ignore[arg-type]

    sent = [item["text"] for item in ctx.bot.sent]
    assert "".join(sent[:2]) == first
    assert sent[2] == "Second"
    assert ctx.bot.events[:2] == [
        ("message", first[:MAX_TELEGRAM_TEXT_LENGTH]),
        ("message", first[MAX_TELEGRAM_TEXT_LENGTH:]),
    ]
    assert ctx.bot.events[2] == ("action", ChatAction.TYPING)
    assert ctx.bot.events[3] == ("sleep", DEFAULT_HANDOFF_DELAY_SECONDS)
    assert sleeper.calls == [DEFAULT_HANDOFF_DELAY_SECONDS]


@pytest.mark.parametrize(
    "update",
    [
        FakeUpdate(),
        FakeUpdate(effective_chat=FakeChat(), effective_message=None),
        private_update(None),
        FakeUpdate(
            effective_chat=FakeChat(type=ChatType.CHANNEL),
            effective_message=FakeTelegramMessage("Hello"),
        ),
        private_update("Hello", chat_id=999),
    ],
)
def test_unsupported_updates_are_ignored(update: FakeUpdate) -> None:
    service = RecordingService()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service,
        ImmediateRuntimeQueue(),
        allowed_chat_ids=ALLOWED,
        handoff_delay=0,
    )
    ctx = context()

    asyncio.run(adapter.handle_text(update, ctx))  # type: ignore[arg-type]

    assert service.calls == []
    assert ctx.bot.sent == []


@pytest.mark.parametrize("chat_type", [ChatType.PRIVATE, ChatType.GROUP, ChatType.SUPERGROUP])
def test_allowlisted_supported_chat_types_enter_shared_queue(chat_type: str) -> None:
    chat_id = 123 if chat_type == ChatType.PRIVATE else -100
    service = RecordingService((assistant("Talker response"),))
    queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service, queue, allowed_chat_ids=ALLOWED, handoff_delay=0
    )
    update = FakeUpdate(
        effective_chat=FakeChat(chat_id, chat_type),
        effective_message=FakeTelegramMessage("Hello"),
    )

    asyncio.run(adapter.handle_text(update, context()))  # type: ignore[arg-type]

    assert service.calls == [(conversation_id_for_chat(chat_id), "Hello")]
    assert len(queue.submissions) == 1
    assert queue.submissions[0]["source"] is RuntimeJobSource.TELEGRAM


def test_topics_use_isolated_ids_and_replies_return_to_originating_topic() -> None:
    service = RecordingService((assistant("Topic response"),))
    queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service, queue, allowed_chat_ids=ALLOWED, handoff_delay=0
    )
    ctx = context()

    for topic in (7, 8):
        update = FakeUpdate(
            effective_chat=FakeChat(-100, ChatType.SUPERGROUP),
            effective_message=FakeTelegramMessage("Hello", message_thread_id=topic),
        )
        asyncio.run(adapter.handle_text(update, ctx))  # type: ignore[arg-type]

    assert service.calls == [
        ("telegram-chat-n100-thread-7", "Hello"),
        ("telegram-chat-n100-thread-8", "Hello"),
    ]
    assert [item["message_thread_id"] for item in ctx.bot.sent] == [7, 8]
    assert len(queue.submissions) == 2


def test_two_groups_and_non_topic_supergroup_messages_use_isolated_group_ids() -> None:
    service = RecordingService((assistant("Group response"),))
    queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service, queue, allowed_chat_ids=ALLOWED, handoff_delay=0
    )

    for chat_id, chat_type in ((-100, ChatType.GROUP), (-200, ChatType.SUPERGROUP)):
        update = FakeUpdate(
            effective_chat=FakeChat(chat_id, chat_type),
            effective_message=FakeTelegramMessage("Hello", message_thread_id=None),
        )
        asyncio.run(adapter.handle_text(update, context()))  # type: ignore[arg-type]

    assert service.calls == [
        ("telegram-chat-n100", "Hello"),
        ("telegram-chat-n200", "Hello"),
    ]
    assert len(queue.submissions) == 2


@pytest.mark.parametrize(
    ("chat_id", "chat_type"),
    [(999, ChatType.PRIVATE), (-999, ChatType.GROUP), (-999, ChatType.SUPERGROUP)],
)
def test_non_allowlisted_chats_are_rejected(chat_id: int, chat_type: str) -> None:
    service = RecordingService((assistant("must not run"),))
    queue = ImmediateRuntimeQueue()
    adapter = TelegramAdapter(service, queue, allowed_chat_ids=ALLOWED)  # type: ignore[arg-type]
    update = FakeUpdate(
        effective_chat=FakeChat(chat_id, chat_type),
        effective_message=FakeTelegramMessage("Hello"),
    )

    asyncio.run(adapter.handle_text(update, context()))  # type: ignore[arg-type]

    assert service.calls == []
    assert queue.submissions == []


def test_service_failure_sends_and_logs_only_sanitized_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="editorial_team.live_trace")

    class FailingService:
        def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
            raise RuntimeError("provider secret and full draft diagnostics")

    adapter = TelegramAdapter(  # type: ignore[arg-type]
        FailingService(),
        ImmediateRuntimeQueue(),
        allowed_chat_ids=ALLOWED,
        handoff_delay=0,
    )
    ctx = context()

    asyncio.run(adapter.handle_text(private_update(), ctx))  # type: ignore[arg-type]

    assert ctx.bot.sent == [{"chat_id": 123, "text": GENERIC_TURN_ERROR}]
    assert "secret" not in str(ctx.bot.sent)
    assert "diagnostics" not in str(ctx.bot.sent)
    assert "update_id=99" in caplog.text
    assert "error_category=runtime_error" in caplog.text
    assert "secret" not in caplog.text
    assert "full draft" not in caplog.text


def test_start_is_platform_onboarding_only() -> None:
    service = RecordingService()
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service, ImmediateRuntimeQueue(), allowed_chat_ids=ALLOWED
    )
    ctx = context()

    asyncio.run(adapter.start(private_update("/start"), ctx))  # type: ignore[arg-type]

    assert ctx.bot.sent == [{"chat_id": 123, "text": ONBOARDING_MESSAGE}]
    assert ONBOARDING_MESSAGE.startswith("Talker\n\n")
    assert service.calls == []


@pytest.mark.parametrize("value", ["", " ", None, 42])
def test_chunking_rejects_empty_or_invalid_text(value: object) -> None:
    with pytest.raises(ValueError):
        chunk_text(value)  # type: ignore[arg-type]


def test_chunking_below_and_at_limit() -> None:
    below = "A" * (MAX_TELEGRAM_TEXT_LENGTH - 1)
    exact = "B" * MAX_TELEGRAM_TEXT_LENGTH

    assert chunk_text(below) == (below,)
    assert chunk_text(exact) == (exact,)


def test_chunking_prefers_paragraph_then_line_boundaries_without_loss() -> None:
    text = "First paragraph.\n\nSecond paragraph that is longer."
    chunks = chunk_text(text, limit=25)

    assert chunks[0] == "First paragraph.\n\n"
    assert "".join(chunks) == text
    assert all(chunks)
    assert all(len(chunk) <= 25 for chunk in chunks)

    line_text = "First line\nSecond line is longer"
    line_chunks = chunk_text(line_text, limit=15)
    assert line_chunks[0] == "First line\n"
    assert "".join(line_chunks) == line_text


def test_chunking_falls_back_to_hard_split_without_character_loss() -> None:
    text = "X" * 25
    chunks = chunk_text(text, limit=10)

    assert chunks == ("X" * 10, "X" * 10, "X" * 5)
    assert "".join(chunks) == text


def test_two_updates_are_serialized_in_processing_order() -> None:
    class BlockingService:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.first_entered = threading.Event()
            self.release_first = threading.Event()
            self.active = 0
            self.maximum_active = 0

        def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append(text)
            if text == "first":
                self.first_entered.set()
                self.release_first.wait(timeout=2)
            self.active -= 1
            return (assistant(f"reply-{text}"),)

    async def scenario() -> tuple[BlockingService, FakeBot]:
        service = BlockingService()
        runtime_queue = RuntimeQueue()
        await runtime_queue.start()
        adapter = TelegramAdapter(  # type: ignore[arg-type]
            service,
            runtime_queue,
            allowed_chat_ids=ALLOWED,
            handoff_delay=0,
        )
        ctx = context()
        first = asyncio.create_task(
            adapter.handle_text(private_update("first"), ctx)  # type: ignore[arg-type]
        )
        entered = await asyncio.to_thread(service.first_entered.wait, 1)
        assert entered
        second = asyncio.create_task(
            adapter.handle_text(private_update("second"), ctx)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        assert service.calls == ["first"]
        service.release_first.set()
        await asyncio.gather(first, second)
        await runtime_queue.close()
        return service, ctx.bot

    service, bot = asyncio.run(scenario())

    assert service.calls == ["first", "second"]
    assert service.maximum_active == 1
    assert [item["text"] for item in bot.sent] == ["reply-first", "reply-second"]
