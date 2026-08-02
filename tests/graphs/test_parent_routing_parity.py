"""Parity tests for implemented parent routing graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from editorial_team.conversation.service import (
    ConversationService,
    ConversationServiceError,
)
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.graphs import build_parent_graph

NOW = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)


def pass_result() -> EditorialResult:
    draft = "Completed draft"
    return EditorialResult(
        writer_output=draft,
        critic_report=CriticReport(CriticVerdict.PASS, "Approved."),
        working_draft=draft,
        revision_applied=False,
    )


def active_state() -> ConversationState:
    report = CriticReport(CriticVerdict.PASS, "Approved.")
    return ConversationState(
        "conversation-1",
        active_task=WritingTask(
            id="task-1",
            conversation_id="conversation-1",
            brief=WritingBrief("Original request"),
            status=WritingTaskStatus.REVIEWED,
            created_at=NOW,
            updated_at=NOW,
            working_draft="Existing draft",
            critic_report=report,
        ),
    )


@dataclass
class RecordingCoordinator:
    output: object
    calls: list[tuple[ConversationState, Message]] = field(default_factory=list)

    def decide(self, state: ConversationState, user_message: Message) -> Any:
        self.calls.append((state, user_message))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingTalker:
    output: object = "Exact Talker response"
    calls: list[tuple[ConversationState, Message]] = field(default_factory=list)

    def respond(self, state: ConversationState, user_message: Message) -> Any:
        self.calls.append((state, user_message))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingStore:
    state: ConversationState | None = None
    loads: list[str] = field(default_factory=list)
    saves: list[ConversationState] = field(default_factory=list)

    def load(self, conversation_id: str) -> ConversationState | None:
        self.loads.append(conversation_id)
        return self.state

    def save(self, state: ConversationState) -> None:
        self.saves.append(state)


@dataclass
class RecordingWorkflow:
    calls: list[WritingTask] = field(default_factory=list)

    def execute(self, task: WritingTask) -> EditorialResult:
        self.calls.append(task)
        return pass_result()


class UnusedWriter:
    def write(self, task: WritingTask) -> str:
        del task
        raise AssertionError("interrupted routing graph must not call Writer")


class UnusedCritic:
    def review(self, task: WritingTask, draft: str) -> CriticReport:
        del task, draft
        raise AssertionError("interrupted routing graph must not call Critic")


class UnusedEditor:
    def revise(
        self,
        task: WritingTask,
        draft: str,
        report: CriticReport,
    ) -> str:
        del task, draft, report
        raise AssertionError("interrupted routing graph must not call Editor")


class FixedIds:
    def __init__(self) -> None:
        self._index = 0

    def __call__(self) -> str:
        self._index += 1
        return f"generated-{self._index}"


@dataclass(frozen=True)
class Harness:
    service: ConversationService
    coordinator: RecordingCoordinator
    talker: RecordingTalker
    store: RecordingStore
    ids: FixedIds


def harness(
    *,
    coordinator_output: object,
    talker_output: object = "Exact Talker response",
    state: ConversationState | None = None,
) -> Harness:
    coordinator = RecordingCoordinator(coordinator_output)
    talker = RecordingTalker(talker_output)
    store = RecordingStore(state)
    ids = FixedIds()
    return Harness(
        service=ConversationService(
            coordinator=coordinator,
            talker=talker,
            workflow=RecordingWorkflow(),
            store=store,
            identifier_generator=ids,
            clock=lambda: NOW,
            max_recent_messages=20,
        ),
        coordinator=coordinator,
        talker=talker,
        store=store,
        ids=ids,
    )


def invoke_graph(
    setup: Harness,
    *,
    interrupt_after: str,
    conversation_id: object = "conversation-1",
    text: object = "User input",
) -> dict[str, object]:
    graph = build_parent_graph(
        coordinator=setup.coordinator,
        talker=setup.talker,
        store=setup.store,
        identifier_generator=setup.ids,
        clock=lambda: NOW,
        writer=UnusedWriter(),
        critic=UnusedCritic(),
        editor=UnusedEditor(),
        max_recent_messages=20,
    ).compile(interrupt_after=[interrupt_after])
    return graph.invoke(
        {
            "state_version": 1,
            "invocation_kind": "conversation",
            "conversation_id": conversation_id,
            "input_text": text,
        }
    )


@pytest.mark.parametrize(
    ("decision", "state"),
    [
        (CoordinatorDecision(CoordinatorRoute.CHAT, 1.0), None),
        (
            CoordinatorDecision(
                CoordinatorRoute.START_WRITING_TASK,
                0.9,
                task_input="Write an announcement.",
            ),
            None,
        ),
        (
            CoordinatorDecision(
                CoordinatorRoute.REVISE_TASK,
                0.8,
                revision_instructions="Make it shorter.",
            ),
            active_state(),
        ),
    ],
)
def test_coordinator_decision_and_inputs_match_conversation_service(
    decision: CoordinatorDecision,
    state: ConversationState | None,
) -> None:
    service_setup = harness(coordinator_output=decision, state=state)
    graph_setup = harness(coordinator_output=decision, state=state)

    service_setup.service.process_message("conversation-1", "User input")
    graph_state = invoke_graph(graph_setup, interrupt_after="coordinator")

    assert graph_state["decision"] == decision
    assert graph_setup.coordinator.calls == service_setup.coordinator.calls
    assert len(graph_setup.coordinator.calls) == 1
    assert graph_setup.talker.calls == []
    assert graph_setup.store.loads == service_setup.store.loads == ["conversation-1"]


def test_talker_response_and_inputs_match_conversation_service() -> None:
    decision = CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    service_setup = harness(coordinator_output=decision)
    graph_setup = harness(coordinator_output=decision)

    service_messages = service_setup.service.process_message(
        "conversation-1",
        "  User input  ",
    )
    graph_state = invoke_graph(
        graph_setup,
        interrupt_after="talker",
        text="  User input  ",
    )

    assert graph_state["talker_response"] == "Exact Talker response"
    assert service_messages[0].content.endswith("Exact Talker response")
    assert graph_setup.coordinator.calls == service_setup.coordinator.calls
    assert graph_setup.talker.calls == service_setup.talker.calls
    assert len(graph_setup.talker.calls) == 1


@pytest.mark.parametrize(
    ("conversation_id", "text"),
    [
        ("", "Input"),
        ("../unsafe", "Input"),
        ("conversation-1", " "),
        (42, "Input"),
        ("conversation-1", None),
    ],
)
def test_input_validation_matches_conversation_service(
    conversation_id: object,
    text: object,
) -> None:
    decision = CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    service_setup = harness(coordinator_output=decision)
    graph_setup = harness(coordinator_output=decision)

    with pytest.raises(ConversationServiceError) as service_error:
        service_setup.service.process_message(conversation_id, text)  # type: ignore[arg-type]
    with pytest.raises(ConversationServiceError) as graph_error:
        invoke_graph(
            graph_setup,
            interrupt_after="coordinator",
            conversation_id=conversation_id,
            text=text,
        )

    assert str(graph_error.value) == str(service_error.value)
    assert str(graph_error.value) == "Invalid conversation input"
    assert graph_setup.coordinator.calls == service_setup.coordinator.calls == []


def test_coordinator_failure_is_sanitized_identically() -> None:
    failure = RuntimeError("private coordinator diagnostics")
    service_setup = harness(coordinator_output=failure)
    graph_setup = harness(coordinator_output=failure)

    with pytest.raises(ConversationServiceError) as service_error:
        service_setup.service.process_message("conversation-1", "Input")
    with pytest.raises(ConversationServiceError) as graph_error:
        invoke_graph(graph_setup, interrupt_after="coordinator", text="Input")

    assert str(graph_error.value) == str(service_error.value) == "Coordinator failed"
    assert graph_error.value.__cause__ is service_error.value.__cause__ is None
    assert "private" not in str(graph_error.value).lower()


@pytest.mark.parametrize("coordinator_output", [object(), "not a decision"])
def test_invalid_coordinator_output_is_rejected_identically(
    coordinator_output: object,
) -> None:
    service_setup = harness(coordinator_output=coordinator_output)
    graph_setup = harness(coordinator_output=coordinator_output)

    with pytest.raises(ConversationServiceError) as service_error:
        service_setup.service.process_message("conversation-1", "Input")
    with pytest.raises(ConversationServiceError) as graph_error:
        invoke_graph(graph_setup, interrupt_after="coordinator", text="Input")

    assert str(graph_error.value) == str(service_error.value)
    assert str(graph_error.value) == "Coordinator returned an invalid decision"
    assert len(graph_setup.coordinator.calls) == len(service_setup.coordinator.calls) == 1


@pytest.mark.parametrize(
    ("talker_output", "message"),
    [
        (RuntimeError("private talker diagnostics"), "Talker failed"),
        (" ", "Talker returned an invalid response"),
    ],
)
def test_talker_failure_is_sanitized_identically(
    talker_output: object,
    message: str,
) -> None:
    decision = CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    service_setup = harness(
        coordinator_output=decision,
        talker_output=talker_output,
    )
    graph_setup = harness(
        coordinator_output=decision,
        talker_output=talker_output,
    )

    with pytest.raises(ConversationServiceError) as service_error:
        service_setup.service.process_message("conversation-1", "Input")
    with pytest.raises(ConversationServiceError) as graph_error:
        invoke_graph(graph_setup, interrupt_after="talker", text="Input")

    assert str(graph_error.value) == str(service_error.value) == message
    assert graph_error.value.__cause__ is service_error.value.__cause__ is None
    assert "private" not in str(graph_error.value).lower()
