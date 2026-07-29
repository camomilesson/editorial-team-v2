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
    DEFAULT_HANDOFF_DELAY_SECONDS,
    GENERIC_TURN_ERROR,
    MAX_TELEGRAM_TEXT_LENGTH,
    ONBOARDING_MESSAGE,
    TelegramAdapter,
    chunk_text,
    conversation_id_for_chat,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


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

    with pytest.raises(ValueError):
        conversation_id_for_chat(True)


@pytest.mark.parametrize("delay", [-1, math.nan, math.inf, True, "slow"])
def test_handoff_delay_must_be_a_non_negative_finite_number(delay: object) -> None:
    with pytest.raises(ValueError, match="handoff_delay"):
        TelegramAdapter(RecordingService(), handoff_delay=delay)  # type: ignore[arg-type]


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
    adapter = TelegramAdapter(  # type: ignore[arg-type]
        service,
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


def test_one_talker_message_has_no_handoff_delay() -> None:
    service = RecordingService((assistant("Talker response"),))
    ctx = context()
    sleeper = RecordingSleeper(ctx.bot.events)
    adapter = TelegramAdapter(service, sleeper=sleeper)  # type: ignore[arg-type]

    asyncio.run(adapter.handle_text(private_update(), ctx))  # type: ignore[arg-type]

    assert [item["text"] for item in ctx.bot.sent] == ["Talker response"]
    assert ctx.bot.actions == []
    assert sleeper.calls == []


def test_chunks_preserve_order_across_application_messages() -> None:
    first = "A" * (MAX_TELEGRAM_TEXT_LENGTH + 2)
    service = RecordingService((assistant(first, 1), assistant("Second", 2)))
    ctx = context()
    sleeper = RecordingSleeper(ctx.bot.events)
    adapter = TelegramAdapter(service, sleeper=sleeper)  # type: ignore[arg-type]

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
            effective_chat=FakeChat(type=ChatType.GROUP),
            effective_message=FakeTelegramMessage("Hello"),
        ),
    ],
)
def test_unsupported_updates_are_ignored(update: FakeUpdate) -> None:
    service = RecordingService()
    adapter = TelegramAdapter(service, handoff_delay=0)  # type: ignore[arg-type]
    ctx = context()

    asyncio.run(adapter.handle_text(update, ctx))  # type: ignore[arg-type]

    assert service.calls == []
    assert ctx.bot.sent == []


def test_service_failure_sends_and_logs_only_sanitized_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="editorial_team.live_trace")

    class FailingService:
        def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
            raise RuntimeError("provider secret and full draft diagnostics")

    adapter = TelegramAdapter(FailingService(), handoff_delay=0)  # type: ignore[arg-type]
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
    adapter = TelegramAdapter(service)  # type: ignore[arg-type]
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
        adapter = TelegramAdapter(service, handoff_delay=0)  # type: ignore[arg-type]
        ctx = context()
        first = asyncio.create_task(
            adapter.handle_text(private_update("first"), ctx)  # type: ignore[arg-type]
        )
        entered = await asyncio.to_thread(service.first_entered.wait, 1)
        assert entered
        second = asyncio.create_task(
            adapter.handle_text(private_update("second"), ctx)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0.05)
        assert service.calls == ["first"]
        service.release_first.set()
        await asyncio.gather(first, second)
        return service, ctx.bot

    service, bot = asyncio.run(scenario())

    assert service.calls == ["first", "second"]
    assert service.maximum_active == 1
    assert [item["text"] for item in bot.sent] == ["reply-first", "reply-second"]
