from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from editorial_team.domain.conversation import (
    ConversationState,
    ConversationStatus,
    Message,
    MessageRole,
)
from editorial_team.domain.editorial import WritingBrief, WritingTask, WritingTaskStatus

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def make_task(conversation_id: str = "conversation-1") -> WritingTask:
    return WritingTask(
        id="task-1",
        conversation_id=conversation_id,
        brief=WritingBrief("Write an announcement"),
        status=WritingTaskStatus.CREATED,
        created_at=NOW,
        updated_at=NOW,
    )


def test_constructs_every_conversation_model() -> None:
    message = Message(
        id="message-1",
        conversation_id="conversation-1",
        role=MessageRole.USER,
        content="Please write an announcement.",
        created_at=NOW,
    )
    state = ConversationState(
        conversation_id="conversation-1",
        recent_messages=(message,),
        status=ConversationStatus.AWAITING_USER_EVALUATION,
        active_task=make_task(),
    )

    assert message.role.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert state.recent_messages == (message,)
    assert state.active_task is not None


def test_conversation_allows_optional_task_state() -> None:
    without_task = ConversationState("conversation-1")
    with_task = ConversationState("conversation-1", active_task=make_task())

    assert without_task.active_task is None
    assert with_task.active_task.id == "task-1"


def test_conversation_message_defaults_are_isolated_and_immutable() -> None:
    first = ConversationState("conversation-1")
    second = ConversationState("conversation-2")

    assert first.recent_messages == second.recent_messages == ()
    assert first.recent_messages is second.recent_messages
    with pytest.raises(FrozenInstanceError):
        first.recent_messages = ()  # type: ignore[misc]


@pytest.mark.parametrize("field", ["id", "conversation_id", "content"])
def test_message_rejects_blank_required_fields(field: str) -> None:
    values = {
        "id": "message-1",
        "conversation_id": "conversation-1",
        "role": MessageRole.USER,
        "content": "Hello",
        "created_at": NOW,
    }
    values[field] = " "

    with pytest.raises(ValueError, match=field):
        Message(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("identifier", ["../message", "has space"])
def test_message_rejects_invalid_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match="id"):
        Message(identifier, "conversation-1", MessageRole.USER, "Hello", NOW)


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 7, 28), datetime(2026, 7, 28, tzinfo=timezone(timedelta(hours=2)))],
)
def test_message_requires_utc_timestamp(timestamp: datetime) -> None:
    with pytest.raises(ValueError, match="created_at"):
        Message("message-1", "conversation-1", MessageRole.USER, "Hello", timestamp)


def test_conversation_rejects_content_from_another_conversation() -> None:
    message = Message("message-1", "conversation-2", MessageRole.USER, "Hello", NOW)

    with pytest.raises(ValueError, match="recent_messages"):
        ConversationState("conversation-1", recent_messages=(message,))

    with pytest.raises(ValueError, match="active_task"):
        ConversationState("conversation-1", active_task=make_task("conversation-2"))
