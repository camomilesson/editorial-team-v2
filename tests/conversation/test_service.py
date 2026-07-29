from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from editorial_team.conversation.formatting import (
    format_critic_report,
    format_editor_message,
    format_talker_message,
    format_writer_message,
)
from editorial_team.conversation.service import ConversationService, ConversationServiceError
from editorial_team.conversation.store import InMemoryConversationStateStore
from editorial_team.domain.conversation import (
    ConversationState,
    Message,
    MessageRole,
)
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute

BASE_TIME = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)


def pass_report() -> CriticReport:
    return CriticReport(CriticVerdict.PASS, "The copy meets the brief.")


def revise_report() -> CriticReport:
    return CriticReport(
        CriticVerdict.REVISE,
        "One change is needed.",
        (
            CriticIssue(
                CriticIssueSeverity.MAJOR,
                "The opening is vague.",
                location="Opening",
                suggestion="Name the benefit.",
                grounded_excerpt="Something new.",
            ),
        ),
    )


def pass_result(output: str = "Writer copy") -> EditorialResult:
    return EditorialResult(output, pass_report(), output, False)


def revise_result() -> EditorialResult:
    return EditorialResult("Writer copy", revise_report(), "Editor copy", True)


def awaiting_task(
    *,
    task_id: str = "task-existing",
    instructions: tuple[str, ...] = ("Keep it concise.",),
) -> WritingTask:
    return WritingTask(
        id=task_id,
        conversation_id="conversation-1",
        brief=WritingBrief("Original request", instructions),
        status=WritingTaskStatus.REVIEWED,
        created_at=BASE_TIME,
        updated_at=BASE_TIME,
        working_draft="Existing copy",
        critic_report=pass_report(),
    )


def awaiting_state(*, task: WritingTask | None = None) -> ConversationState:
    return ConversationState(
        "conversation-1",
        active_task=task or awaiting_task(),
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
    output: object = "Acknowledged."
    calls: list[tuple[ConversationState, Message]] = field(default_factory=list)

    def respond(self, state: ConversationState, user_message: Message) -> Any:
        self.calls.append((state, user_message))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingWorkflow:
    output: object = field(default_factory=pass_result)
    calls: list[WritingTask] = field(default_factory=list)

    def execute(self, task: WritingTask) -> Any:
        self.calls.append(task)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class RecordingStore:
    def __init__(self, state: ConversationState | None = None) -> None:
        self.state = state
        self.loads: list[str] = []
        self.saves: list[ConversationState] = []

    def load(self, conversation_id: str) -> ConversationState | None:
        self.loads.append(conversation_id)
        return self.state

    def save(self, state: ConversationState) -> None:
        self.saves.append(state)
        self.state = state


class SequenceIds:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> str:
        self.count += 1
        return f"generated-{self.count}"


class SequenceClock:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> datetime:
        value = BASE_TIME + timedelta(seconds=self.count)
        self.count += 1
        return value


def make_service(
    decision: CoordinatorDecision,
    *,
    state: ConversationState | None = None,
    talker_output: object = "Acknowledged.",
    workflow_output: object | None = None,
    max_messages: int = 20,
) -> tuple[
    ConversationService,
    RecordingCoordinator,
    RecordingTalker,
    RecordingWorkflow,
    RecordingStore,
]:
    coordinator = RecordingCoordinator(decision)
    talker = RecordingTalker(talker_output)
    workflow = RecordingWorkflow(
        pass_result() if workflow_output is None else workflow_output
    )
    store = RecordingStore(state)
    service = ConversationService(
        coordinator=coordinator,
        talker=talker,
        workflow=workflow,
        store=store,
        identifier_generator=SequenceIds(),
        clock=SequenceClock(),
        max_recent_messages=max_messages,
    )
    return service, coordinator, talker, workflow, store


def test_chat_creates_conversation_and_saves_once() -> None:
    service, coordinator, talker, workflow, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker_output="Hello there.",
    )

    returned = service.process_message("conversation-1", "Hello")

    assert store.loads == ["conversation-1"]
    assert len(store.saves) == 1
    assert len(returned) == 1
    assert returned[0].content == "Talker\n\nHello there."
    assert returned[0].role is MessageRole.ASSISTANT
    assert returned[0].id == "generated-2"
    assert returned[0].created_at == BASE_TIME + timedelta(seconds=1)
    saved = store.saves[0]
    assert [message.role for message in saved.recent_messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert saved.recent_messages[0].id == "generated-1"
    assert saved.recent_messages[0].created_at == BASE_TIME
    assert coordinator.calls[0][0].recent_messages[-1] is coordinator.calls[0][1]
    assert talker.calls == [coordinator.calls[0]]
    assert workflow.calls == []


@pytest.mark.parametrize(
    ("conversation_id", "text"),
    [("", "Hello"), ("../bad", "Hello"), ("conversation-1", ""), ("conversation-1", 42)],
)
def test_invalid_user_input_fails_before_load_or_save(
    conversation_id: str,
    text: object,
) -> None:
    service, _, _, _, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
    )

    with pytest.raises(ConversationServiceError, match="Invalid conversation input"):
        service.process_message(conversation_id, text)  # type: ignore[arg-type]

    assert store.loads == []
    assert store.saves == []


def test_chat_continues_existing_state_and_preserves_latest_task() -> None:
    original = awaiting_state()
    service, _, talker, _, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 0.8),
        state=original,
    )

    service.process_message("conversation-1", "What is the weather like?")

    saved = store.saves[0]
    assert saved.active_task is original.active_task
    assert talker.calls[0][0].active_task is original.active_task
    assert original.recent_messages == ()


@pytest.mark.parametrize("output", ["", " ", None, 42])
def test_chat_rejects_invalid_talker_output_without_save(output: object) -> None:
    service, _, _, _, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker_output=output,
    )

    with pytest.raises(ConversationServiceError, match="invalid response"):
        service.process_message("conversation-1", "Hello")

    assert store.saves == []


def test_start_pass_creates_task_and_orders_messages() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        0.9,
        task_input="Write a launch post.",
    )
    result = pass_result("Exact Writer output")
    service, _, talker, workflow, store = make_service(
        decision,
        workflow_output=result,
    )

    returned = service.process_message("conversation-1", "Please write it")

    assert talker.calls == []
    assert len(workflow.calls) == 1
    workflow_task = workflow.calls[0]
    assert workflow_task.id == "generated-2"
    assert workflow_task.brief == WritingBrief("Write a launch post.")
    assert workflow_task.status is WritingTaskStatus.CREATED
    assert workflow_task.working_draft is None
    assert [message.content for message in returned] == [
        format_writer_message(result.writer_output),
        format_critic_report(result.critic_report),
        format_editor_message(result),
    ]
    saved = store.saves[0]
    assert saved.active_task is not None
    assert saved.active_task.id == workflow_task.id
    assert saved.active_task.working_draft == "Exact Writer output"
    assert saved.active_task.critic_report is result.critic_report
    assert saved.active_task.status is WritingTaskStatus.REVIEWED


def test_start_revise_distinguishes_writer_and_editor_outputs() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        0.9,
        task_input="Write copy.",
    )
    service, _, _, _, store = make_service(decision, workflow_output=revise_result())

    returned = service.process_message("conversation-1", "Write it")

    assert returned[0].content == "Writer\n\nWriter copy"
    assert returned[2].content == "Editor\n\nEditor copy"
    assert len(returned) == 3
    assert store.saves[0].active_task.working_draft == "Editor copy"  # type: ignore[union-attr]


def test_start_replaces_previous_active_task() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="A new request",
    )
    service, _, _, _, store = make_service(decision, state=awaiting_state())

    service.process_message("conversation-1", "Start something new")

    assert store.saves[0].active_task.id == "generated-2"  # type: ignore[union-attr]


@pytest.mark.parametrize("result", [pass_result("New Writer copy"), revise_result()])
def test_revision_runs_normal_workflow_and_replaces_canonical_state(
    result: EditorialResult,
) -> None:
    original = awaiting_state()
    decision = CoordinatorDecision(
        CoordinatorRoute.REVISE_TASK,
        0.9,
        revision_instructions="Use a warmer ending.",
    )
    service, _, _, workflow, store = make_service(
        decision,
        state=original,
        workflow_output=result,
    )

    returned = service.process_message("conversation-1", "Make the ending warmer")

    assert len(workflow.calls) == 1
    workflow_task = workflow.calls[0]
    old_task = original.active_task
    assert old_task is not None
    assert workflow_task.id == old_task.id
    assert workflow_task.conversation_id == old_task.conversation_id
    assert workflow_task.created_at == old_task.created_at
    assert workflow_task.working_draft == "Existing copy"
    assert workflow_task.brief.original_request == "Original request"
    assert workflow_task.brief.instructions == (
        "Keep it concise.",
        "Use a warmer ending.",
    )
    assert old_task.brief.instructions == ("Keep it concise.",)
    saved_task = store.saves[0].active_task
    assert saved_task is not None
    assert saved_task.id == old_task.id
    assert saved_task.working_draft == result.working_draft
    assert saved_task.critic_report is result.critic_report
    expected_status = (
        WritingTaskStatus.REVISED
        if result.revision_applied
        else WritingTaskStatus.REVIEWED
    )
    assert saved_task.status is expected_status
    assert len(returned) == 3
    assert returned[0].content == format_writer_message(result.writer_output)
    assert returned[2].content == format_editor_message(result)


def test_revision_without_active_task_fails_without_save() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.REVISE_TASK,
        1.0,
        revision_instructions="Change it.",
    )
    service, _, _, workflow, store = make_service(decision)

    with pytest.raises(ConversationServiceError, match="No writing task"):
        service.process_message("conversation-1", "Change it")

    assert workflow.calls == []
    assert store.saves == []


def test_revision_without_working_draft_fails_without_save() -> None:
    task = awaiting_task()
    object.__setattr__(task, "working_draft", None)
    state = awaiting_state(task=task)
    decision = CoordinatorDecision(
        CoordinatorRoute.REVISE_TASK,
        1.0,
        revision_instructions="Change it.",
    )
    service, _, _, workflow, store = make_service(decision, state=state)

    with pytest.raises(ConversationServiceError, match="No writing task"):
        service.process_message("conversation-1", "Change it")

    assert workflow.calls == []
    assert store.saves == []


def test_malformed_coordinator_decision_is_rejected_without_save() -> None:
    decision = CoordinatorDecision(
        CoordinatorRoute.REVISE_TASK,
        1.0,
        revision_instructions="Change it.",
    )
    object.__setattr__(decision, "revision_instructions", " ")
    service, _, _, workflow, store = make_service(decision, state=awaiting_state())

    with pytest.raises(ConversationServiceError, match="invalid decision"):
        service.process_message("conversation-1", "Change it")

    assert workflow.calls == []
    assert store.saves == []


def test_critic_formatting_is_deterministic_and_omits_absent_optional_fields() -> None:
    full = format_critic_report(revise_report())
    assert full == (
        "Critic\n\n"
        "Verdict: REVISE\n\n"
        "Summary: One change is needed.\n\n"
        "Issues:\n\n"
        "1. Severity: MAJOR\n\n"
        "Location: Opening\n\n"
        "Problem: The opening is vague.\n\n"
        "Suggestion: Name the benefit.\n\n"
        "Grounded excerpt: Something new."
    )
    minimal = CriticReport(
        CriticVerdict.REVISE,
        "Fix one issue.",
        (CriticIssue(CriticIssueSeverity.MINOR, "Typo."),),
    )
    assert format_critic_report(minimal) == (
        "Critic\n\n"
        "Verdict: REVISE\n\n"
        "Summary: Fix one issue.\n\n"
        "Issues:\n\n"
        "1. Severity: MINOR\n\n"
        "Problem: Typo."
    )
    assert format_critic_report(pass_report()).endswith("Issues: None")


def test_critic_formatting_preserves_multiple_issue_order() -> None:
    report = CriticReport(
        CriticVerdict.REVISE,
        "Two changes are needed.",
        (
            CriticIssue(CriticIssueSeverity.MAJOR, "First problem."),
            CriticIssue(
                CriticIssueSeverity.MINOR,
                "Second problem.",
                suggestion="Fix second.",
            ),
        ),
    )

    formatted = format_critic_report(report)

    assert formatted.index("1. Severity: MAJOR") < formatted.index(
        "2. Severity: MINOR"
    )
    assert "Problem: First problem.\n\n2. Severity: MINOR" in formatted
    assert "Suggestion: Fix second." in formatted
    assert "Location:" not in formatted
    assert "Grounded excerpt:" not in formatted
    assert "None" not in formatted


def test_recent_messages_trim_oldest_without_affecting_task() -> None:
    old_messages = tuple(
        Message(
            f"old-{index}",
            "conversation-1",
            MessageRole.USER,
            f"Old {index}",
            BASE_TIME,
        )
        for index in range(3)
    )
    original = replace(awaiting_state(), recent_messages=old_messages)
    service, _, _, _, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        state=original,
        max_messages=3,
    )

    service.process_message("conversation-1", "New")

    saved = store.saves[0]
    assert [message.content for message in saved.recent_messages] == [
        "Old 2",
        "New",
        format_talker_message("Acknowledged."),
    ]
    assert saved.active_task is original.active_task
    assert len(original.recent_messages) == 3


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_recent_message_limit_must_be_positive_integer(maximum: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ConversationService(
            coordinator=RecordingCoordinator(
                CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
            ),
            talker=RecordingTalker(),
            workflow=RecordingWorkflow(),
            store=RecordingStore(),
            identifier_generator=SequenceIds(),
            clock=SequenceClock(),
            max_recent_messages=maximum,  # type: ignore[arg-type]
        )


def test_store_isolates_conversations_and_uses_defensive_copies() -> None:
    store = InMemoryConversationStateStore()
    one = ConversationState("conversation-1")
    two = ConversationState("conversation-2")
    store.save(one)
    store.save(two)

    loaded_one = store.load("conversation-1")
    loaded_two = store.load("conversation-2")
    assert loaded_one == one
    assert loaded_two == two
    assert loaded_one is not one
    assert loaded_two is not two
    assert store.load("missing") is None

    corrupted = store.load("conversation-1")
    assert corrupted is not None
    object.__setattr__(corrupted, "recent_messages", ())
    assert store.load("conversation-1") == one


def test_store_sharing_is_explicit_between_services() -> None:
    shared = InMemoryConversationStateStore()

    def service_for(store: InMemoryConversationStateStore) -> ConversationService:
        return ConversationService(
            coordinator=RecordingCoordinator(
                CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)
            ),
            talker=RecordingTalker(),
            workflow=RecordingWorkflow(),
            store=store,
            identifier_generator=SequenceIds(),
            clock=SequenceClock(),
            max_recent_messages=20,
        )

    service_for(shared).process_message("conversation-1", "First")
    service_for(shared).process_message("conversation-1", "Second")
    assert len(shared.load("conversation-1").recent_messages) == 4  # type: ignore[union-attr]

    isolated = InMemoryConversationStateStore()
    service_for(isolated).process_message("conversation-1", "Only")
    assert len(isolated.load("conversation-1").recent_messages) == 2  # type: ignore[union-attr]
    assert len(shared.load("conversation-1").recent_messages) == 4  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("decision", "talker_output", "workflow_output", "message"),
    [
        (
            CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
            RuntimeError("secret talker diagnostics"),
            None,
            "Talker failed",
        ),
        (
            CoordinatorDecision(
                CoordinatorRoute.START_WRITING_TASK,
                1.0,
                task_input="Write it",
            ),
            "Okay",
            RuntimeError("secret workflow diagnostics"),
            "Writing workflow failed",
        ),
    ],
)
def test_dependency_failures_are_sanitized_and_never_saved(
    decision: CoordinatorDecision,
    talker_output: object,
    workflow_output: object,
    message: str,
) -> None:
    service, _, _, _, store = make_service(
        decision,
        talker_output=talker_output,
        workflow_output=workflow_output,
    )

    with pytest.raises(ConversationServiceError, match=rf"^{message}$") as caught:
        service.process_message("conversation-1", "Input")

    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert store.saves == []


def test_coordinator_failure_is_sanitized_and_never_saved() -> None:
    coordinator = RecordingCoordinator(RuntimeError("secret coordinator diagnostics"))
    store = RecordingStore()
    service = ConversationService(
        coordinator=coordinator,
        talker=RecordingTalker(),
        workflow=RecordingWorkflow(),
        store=store,
        identifier_generator=SequenceIds(),
        clock=SequenceClock(),
        max_recent_messages=20,
    )

    with pytest.raises(ConversationServiceError, match=r"^Coordinator failed$") as caught:
        service.process_message("conversation-1", "Input")

    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert store.saves == []


def test_invalid_workflow_result_is_rejected_without_save() -> None:
    malformed = pass_result()
    object.__setattr__(malformed, "working_draft", "Different")
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="Write it",
    )
    service, _, _, _, store = make_service(decision, workflow_output=malformed)

    with pytest.raises(ConversationServiceError, match="invalid result"):
        service.process_message("conversation-1", "Input")

    assert store.saves == []


def test_workflow_result_with_malformed_report_is_rejected_without_save() -> None:
    malformed = pass_result()
    object.__setattr__(malformed.critic_report, "summary", " ")
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="Write it",
    )
    service, _, _, _, store = make_service(decision, workflow_output=malformed)

    with pytest.raises(ConversationServiceError, match="invalid result"):
        service.process_message("conversation-1", "Input")

    assert store.saves == []


def test_loaded_state_is_not_mutated_when_route_fails() -> None:
    original = awaiting_state()
    snapshot = deepcopy(original)
    service, _, _, _, store = make_service(
        CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        state=original,
        talker_output=RuntimeError("failure"),
    )

    with pytest.raises(ConversationServiceError):
        service.process_message("conversation-1", "Input")

    assert original == snapshot
    assert store.saves == []


def test_failed_revision_does_not_overwrite_latest_task() -> None:
    original = awaiting_state()
    snapshot = deepcopy(original)
    service, _, _, workflow, store = make_service(
        CoordinatorDecision(
            CoordinatorRoute.REVISE_TASK,
            1.0,
            revision_instructions="Make it shorter.",
        ),
        state=original,
        workflow_output=RuntimeError("failure"),
    )

    with pytest.raises(ConversationServiceError, match="Writing workflow failed"):
        service.process_message("conversation-1", "Make it shorter")

    assert len(workflow.calls) == 1
    assert original == snapshot
    assert store.state == original
    assert store.saves == []
