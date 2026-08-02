"""Integration tests for the ConversationService graph cutover boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from editorial_team.conversation.service import (
    ConversationService,
    ConversationServiceError,
)
from editorial_team.domain.conversation import ConversationState, Message, MessageRole

NOW = datetime(2026, 8, 2, 14, 0, tzinfo=UTC)


class UnusedDependency:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"legacy dependency {name} must not be used")


@dataclass
class StoreSpy:
    saves: list[ConversationState] = field(default_factory=list)

    def load(self, conversation_id: str) -> ConversationState | None:
        del conversation_id
        raise AssertionError("injected graph owns repository loading")

    def save(self, state: ConversationState) -> None:
        self.saves.append(state)


@dataclass
class GraphSpy:
    output: object
    calls: list[tuple[dict[str, object], dict[str, object]]] = field(
        default_factory=list
    )

    def invoke(
        self,
        state: dict[str, object],
        config: dict[str, object],
    ) -> Any:
        self.calls.append((state, config))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def build_service(graph: GraphSpy, store: StoreSpy) -> ConversationService:
    return ConversationService(
        coordinator=UnusedDependency(),  # type: ignore[arg-type]
        talker=UnusedDependency(),  # type: ignore[arg-type]
        workflow=UnusedDependency(),  # type: ignore[arg-type]
        store=store,
        identifier_generator=lambda: "unused-id",
        clock=lambda: NOW,
        max_recent_messages=20,
        graph_runner=graph,
    )


def test_process_message_invokes_graph_and_commits_only_completed_output() -> None:
    assistant = Message(
        "assistant-1",
        "conversation-1",
        MessageRole.ASSISTANT,
        "Exact graph response",
        NOW,
    )
    completed = ConversationState("conversation-1", recent_messages=(assistant,))
    graph = GraphSpy(
        {
            "assistant_messages": (assistant,),
            "completed_conversation": completed,
        }
    )
    store = StoreSpy()

    returned = build_service(graph, store).process_message(
        "conversation-1",
        "User input",
    )

    assert returned == (assistant,)
    assert store.saves == [completed]
    assert graph.calls == [
        (
            {
                "state_version": 1,
                "invocation_kind": "conversation",
                "conversation_id": "conversation-1",
                "input_text": "User input",
            },
            {
                "configurable": {
                    "thread_id": "editorial:v1:conversation-1",
                }
            },
        )
    ]


def test_graph_failure_is_resanitized_and_never_committed() -> None:
    error = ConversationServiceError("Talker failed")
    error.add_note("private graph task metadata")
    graph = GraphSpy(error)
    store = StoreSpy()

    with pytest.raises(ConversationServiceError, match=r"^Talker failed$") as caught:
        build_service(graph, store).process_message("conversation-1", "Input")

    assert caught.value.__cause__ is None
    assert not getattr(caught.value, "__notes__", [])
    assert store.saves == []


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"assistant_messages": (), "completed_conversation": ConversationState("c")},
        {"assistant_messages": ("not-a-message",), "completed_conversation": object()},
    ],
)
def test_invalid_graph_output_is_rejected_without_commit(output: object) -> None:
    graph = GraphSpy(output)
    store = StoreSpy()

    with pytest.raises(
        ConversationServiceError,
        match=r"^Conversation graph returned an invalid result$",
    ):
        build_service(graph, store).process_message("conversation-1", "Input")

    assert store.saves == []
