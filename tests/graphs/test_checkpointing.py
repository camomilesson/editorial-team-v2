"""Tests for the process-local graph checkpointer factory."""

from datetime import UTC, datetime

from langgraph.checkpoint.memory import InMemorySaver

from editorial_team.domain.conversation import ConversationState, Message, MessageRole
from editorial_team.graphs import create_in_memory_checkpointer


def test_factory_returns_fresh_in_memory_checkpointers() -> None:
    first = create_in_memory_checkpointer()
    second = create_in_memory_checkpointer()

    assert isinstance(first, InMemorySaver)
    assert isinstance(second, InMemorySaver)
    assert first is not second


def test_checkpointer_serializer_round_trips_allowed_domain_state() -> None:
    checkpointer = create_in_memory_checkpointer()
    state = ConversationState(
        "conversation-1",
        recent_messages=(
            Message(
                "message-1",
                "conversation-1",
                MessageRole.USER,
                "Hello",
                datetime(2026, 8, 2, tzinfo=UTC),
            ),
        ),
    )

    type_name, payload = checkpointer.serde.dumps_typed(state)
    restored = checkpointer.serde.loads_typed((type_name, payload))

    assert restored == state
