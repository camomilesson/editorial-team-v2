"""Tests for the thin conversation graph facade."""

from datetime import UTC, datetime

import pytest

from editorial_team.conversation import ConversationService, ConversationServiceError
from editorial_team.domain.conversation import Message, MessageRole
from editorial_team.errors import ServiceError

NOW = datetime(2026, 8, 3, tzinfo=UTC)


class Graph:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[tuple[object, object]] = []

    def invoke(self, state: object, config: object) -> object:
        self.calls.append((state, config))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def message() -> Message:
    return Message("message-1", "conversation-1", MessageRole.ASSISTANT, "Reply", NOW)


def test_facade_invokes_stable_thread_and_returns_messages() -> None:
    graph = Graph({"assistant_messages": (message(),)})
    service = ConversationService(graph_runner=graph)

    assert service.process_message("conversation-1", "Hello") == (message(),)
    assert graph.calls == [
        (
            {
                "state_version": 1,
                "invocation_kind": "conversation",
                "conversation_id": "conversation-1",
                "input_text": "Hello",
            },
            {"configurable": {"thread_id": "editorial:v1:conversation-1"}},
        )
    ]


@pytest.mark.parametrize(("conversation_id", "text"), [("", "x"), ("../x", "x"), ("c", " ")])
def test_facade_rejects_invalid_input(conversation_id: str, text: str) -> None:
    with pytest.raises(ConversationServiceError, match="Invalid conversation input"):
        ConversationService(graph_runner=Graph({})).process_message(conversation_id, text)


def test_facade_sanitizes_graph_failures_and_invalid_output() -> None:
    with pytest.raises(ConversationServiceError, match="Talker failed"):
        ConversationService(graph_runner=Graph(ServiceError("Talker failed"))).process_message(
            "conversation-1", "Hello"
        )
    with pytest.raises(ConversationServiceError, match="Conversation graph failed"):
        ConversationService(graph_runner=Graph(RuntimeError("private"))).process_message(
            "conversation-1", "Hello"
        )
    with pytest.raises(ConversationServiceError, match="invalid result"):
        ConversationService(graph_runner=Graph({})).process_message("conversation-1", "Hello")


def test_facade_closes_checkpoint_once() -> None:
    closed: list[bool] = []
    service = ConversationService(
        graph_runner=Graph({}), close_checkpointer=lambda: closed.append(True)
    )
    service.close()
    service.close()
    assert closed == [True]
