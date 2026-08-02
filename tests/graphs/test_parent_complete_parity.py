"""Whole-turn parity tests for the complete disconnected parent graph."""

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
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.graphs import build_parent_graph, create_in_memory_checkpointer
from editorial_team.workflows import WritingWorkflow

NOW = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)


def pass_report() -> CriticReport:
    return CriticReport(CriticVerdict.PASS, "The draft meets the brief.")


def revise_report() -> CriticReport:
    return CriticReport(
        CriticVerdict.REVISE,
        "One revision is required.",
        (
            CriticIssue(
                CriticIssueSeverity.MAJOR,
                "The opening is unclear.",
                suggestion="Name the benefit.",
            ),
        ),
    )


def existing_state(*, recent_messages: tuple[Message, ...] = ()) -> ConversationState:
    return ConversationState(
        "conversation-1",
        recent_messages=recent_messages,
        active_task=WritingTask(
            id="task-existing",
            conversation_id="conversation-1",
            brief=WritingBrief("Original request", ("Keep it concise.",)),
            status=WritingTaskStatus.REVIEWED,
            created_at=NOW,
            updated_at=NOW,
            working_draft="Existing draft",
            critic_report=pass_report(),
        ),
    )


@dataclass
class CoordinatorFake:
    output: CoordinatorDecision
    calls: list[tuple[ConversationState, Message]] = field(default_factory=list)

    def decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        self.calls.append((state, user_message))
        return self.output


@dataclass
class TalkerFake:
    output: object = "Exact Talker response"
    calls: list[tuple[ConversationState, Message]] = field(default_factory=list)

    def respond(self, state: ConversationState, user_message: Message) -> Any:
        self.calls.append((state, user_message))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class WriterFake:
    output: object = "Exact Writer draft"
    calls: list[WritingTask] = field(default_factory=list)

    def write(self, task: WritingTask) -> Any:
        self.calls.append(task)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class CriticFake:
    output: object = field(default_factory=pass_report)
    calls: list[tuple[WritingTask, str]] = field(default_factory=list)

    def review(self, task: WritingTask, draft: str) -> Any:
        self.calls.append((task, draft))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class EditorFake:
    output: object = "Exact Editor draft"
    calls: list[tuple[WritingTask, str, CriticReport]] = field(default_factory=list)

    def revise(self, task: WritingTask, draft: str, report: CriticReport) -> Any:
        self.calls.append((task, draft, report))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class StoreFake:
    state: ConversationState | None = None
    saves: list[ConversationState] = field(default_factory=list)

    def load(self, conversation_id: str) -> ConversationState | None:
        assert conversation_id == "conversation-1"
        return self.state

    def save(self, state: ConversationState) -> None:
        self.saves.append(state)


class SequenceIds:
    def __init__(self) -> None:
        self.index = 0

    def __call__(self) -> str:
        self.index += 1
        return f"generated-{self.index}"


@dataclass(frozen=True)
class Actors:
    coordinator: CoordinatorFake
    talker: TalkerFake
    writer: WriterFake
    critic: CriticFake
    editor: EditorFake
    store: StoreFake
    ids: SequenceIds


def actors(
    decision: CoordinatorDecision,
    *,
    state: ConversationState | None = None,
    talker_output: object = "Exact Talker response",
    writer_output: object = "Exact Writer draft",
    critic_output: object | None = None,
    editor_output: object = "Exact Editor draft",
) -> Actors:
    return Actors(
        coordinator=CoordinatorFake(decision),
        talker=TalkerFake(talker_output),
        writer=WriterFake(writer_output),
        critic=CriticFake(pass_report() if critic_output is None else critic_output),
        editor=EditorFake(editor_output),
        store=StoreFake(state),
        ids=SequenceIds(),
    )


def service_for(setup: Actors, *, max_recent_messages: int = 50) -> ConversationService:
    return ConversationService(
        coordinator=setup.coordinator,
        talker=setup.talker,
        workflow=WritingWorkflow(
            writer=setup.writer,
            critic=setup.critic,
            editor=setup.editor,
        ),
        store=setup.store,
        identifier_generator=setup.ids,
        clock=lambda: NOW,
        max_recent_messages=max_recent_messages,
    )


def graph_for(
    setup: Actors,
    *,
    max_recent_messages: int = 50,
    checkpointer: object | None = None,
) -> Any:
    builder = build_parent_graph(
        coordinator=setup.coordinator,
        talker=setup.talker,
        store=setup.store,
        identifier_generator=setup.ids,
        clock=lambda: NOW,
        writer=setup.writer,
        critic=setup.critic,
        editor=setup.editor,
        max_recent_messages=max_recent_messages,
    )
    return builder.compile(checkpointer=checkpointer)


def invoke_graph(graph: Any, text: str, *, config: dict[str, object] | None = None) -> Any:
    return graph.invoke(
        {
            "state_version": 1,
            "invocation_kind": "conversation",
            "conversation_id": "conversation-1",
            "input_text": text,
        },
        config,
    )


def assert_success_parity(
    service_setup: Actors,
    graph_setup: Actors,
    service_messages: tuple[Message, ...],
    graph_state: dict[str, object],
) -> None:
    assert tuple(graph_state["assistant_messages"]) == service_messages  # type: ignore[arg-type]
    assert graph_state["completed_conversation"] == service_setup.store.saves[0]
    assert graph_setup.store.saves == []
    assert graph_setup.coordinator.calls == service_setup.coordinator.calls
    assert graph_setup.talker.calls == service_setup.talker.calls
    assert graph_setup.writer.calls == service_setup.writer.calls
    assert graph_setup.critic.calls == service_setup.critic.calls
    assert graph_setup.editor.calls == service_setup.editor.calls


def test_complete_chat_turn_matches_service() -> None:
    decision = CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    service_setup = actors(decision)
    graph_setup = actors(decision)

    service_messages = service_for(service_setup).process_message(
        "conversation-1", "Hello"
    )
    graph_state = invoke_graph(graph_for(graph_setup), "Hello")

    assert_success_parity(service_setup, graph_setup, service_messages, graph_state)


@pytest.mark.parametrize("report", [pass_report(), revise_report()])
def test_complete_new_task_matches_service(report: CriticReport) -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        0.9,
        task_input="Write an announcement.",
    )
    service_setup = actors(decision, critic_output=report)
    graph_setup = actors(decision, critic_output=report)

    service_messages = service_for(service_setup).process_message(
        "conversation-1", "Please write it"
    )
    graph_state = invoke_graph(graph_for(graph_setup), "Please write it")

    assert_success_parity(service_setup, graph_setup, service_messages, graph_state)
    assert graph_setup.writer.calls[0] == service_setup.writer.calls[0]
    assert graph_setup.writer.calls[0].id == "generated-2"
    assert graph_setup.writer.calls[0].status is WritingTaskStatus.CREATED
    assert graph_state["editorial_result"].critic_report == report  # type: ignore[union-attr]


def test_complete_revision_matches_service() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.REVISE_TASK,
        0.9,
        revision_instructions="Use a warmer ending.",
    )
    prior = existing_state()
    service_setup = actors(decision, state=prior, critic_output=revise_report())
    graph_setup = actors(decision, state=prior, critic_output=revise_report())

    service_messages = service_for(service_setup).process_message(
        "conversation-1", "Make it warmer"
    )
    graph_state = invoke_graph(graph_for(graph_setup), "Make it warmer")

    assert_success_parity(service_setup, graph_setup, service_messages, graph_state)
    graph_task = graph_setup.writer.calls[0]
    assert graph_task == service_setup.writer.calls[0]
    assert graph_task.id == "task-existing"
    assert graph_task.brief.instructions == (
        "Keep it concise.",
        "Use a warmer ending.",
    )
    assert prior.active_task is not None
    assert prior.active_task.brief.instructions == ("Keep it concise.",)


def test_history_trimming_matches_service() -> None:
    old_messages = tuple(
        Message(
            f"old-{index}",
            "conversation-1",
            MessageRole.USER,
            f"Old {index}",
            NOW,
        )
        for index in range(3)
    )
    decision = CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    prior = existing_state(recent_messages=old_messages)
    service_setup = actors(decision, state=prior)
    graph_setup = actors(decision, state=prior)

    service_messages = service_for(
        service_setup, max_recent_messages=3
    ).process_message("conversation-1", "New input")
    graph_state = invoke_graph(
        graph_for(graph_setup, max_recent_messages=3),
        "New input",
    )

    assert_success_parity(service_setup, graph_setup, service_messages, graph_state)
    completed = graph_state["completed_conversation"]
    assert isinstance(completed, ConversationState)
    assert [message.content for message in completed.recent_messages] == [
        "Old 2",
        "New input",
        service_messages[0].content,
    ]
    assert completed.active_task is prior.active_task


@pytest.mark.parametrize(
    ("writer_output", "critic_output", "editor_output"),
    [
        (RuntimeError("private writer diagnostics"), pass_report(), "Editor"),
        ("Writer", RuntimeError("private critic diagnostics"), "Editor"),
        ("Writer", revise_report(), RuntimeError("private editor diagnostics")),
    ],
)
def test_writing_failure_is_sanitized_and_atomic(
    writer_output: object,
    critic_output: object,
    editor_output: object,
) -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="Write it.",
    )
    service_setup = actors(
        decision,
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )
    graph_setup = actors(
        decision,
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )
    checkpointer = create_in_memory_checkpointer()
    config = {"configurable": {"thread_id": "failure-parity"}}

    with pytest.raises(ConversationServiceError) as service_error:
        service_for(service_setup).process_message("conversation-1", "Write it")
    graph = graph_for(graph_setup, checkpointer=checkpointer)
    with pytest.raises(ConversationServiceError) as graph_error:
        invoke_graph(graph, "Write it", config=config)

    assert str(graph_error.value) == str(service_error.value) == "Writing workflow failed"
    assert graph_error.value.__cause__ is service_error.value.__cause__ is None
    assert "private" not in str(graph_error.value).lower()
    assert service_setup.store.saves == graph_setup.store.saves == []
    assert "completed_conversation" not in graph.get_state(config).values
