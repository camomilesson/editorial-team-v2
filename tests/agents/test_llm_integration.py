from __future__ import annotations

import json
from datetime import UTC, datetime

from editorial_team.agents import (
    LlmCoordinator,
    LlmCritic,
    LlmEditor,
    LlmTalker,
    LlmWriter,
)
from editorial_team.agents.schemas import (
    COORDINATOR_STRUCTURED_OUTPUT,
    CRITIC_STRUCTURED_OUTPUT,
)
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.domain.conversation import ConversationState, ConversationStatus
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.models import FakeModelClient, ModelResponse
from editorial_team.workflows import WritingWorkflow

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def response(text: str) -> ModelResponse:
    return ModelResponse(text, (), None)


def decision(
    route: str,
    *,
    task_input: str | None = None,
    revision_instructions: str | None = None,
) -> ModelResponse:
    return response(
        json.dumps(
            {
                "route": route,
                "confidence": 0.95,
                "task_input": task_input,
                "revision_instructions": revision_instructions,
            }
        )
    )


def report(verdict: str) -> ModelResponse:
    issues = []
    if verdict == "revise":
        issues = [
            {
                "severity": "major",
                "problem": "The opening needs focus.",
                "grounded_excerpt": "Writer output",
            }
        ]
    return response(
        json.dumps(
            {
                "verdict": verdict,
                "summary": "Review complete.",
                "issues": issues,
            }
        )
    )


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"id-{self.value}"


def build_service(
    *,
    coordinator_responses: list[ModelResponse],
    talker_responses: list[ModelResponse] | None = None,
    writer_responses: list[ModelResponse] | None = None,
    critic_responses: list[ModelResponse] | None = None,
    editor_responses: list[ModelResponse] | None = None,
    store: InMemoryConversationStateStore | None = None,
) -> tuple[
    ConversationService,
    InMemoryConversationStateStore,
    FakeModelClient,
    FakeModelClient,
    FakeModelClient,
    FakeModelClient,
    FakeModelClient,
]:
    coordinator_model = FakeModelClient(coordinator_responses)
    talker_model = FakeModelClient(talker_responses or [])
    writer_model = FakeModelClient(writer_responses or [])
    critic_model = FakeModelClient(critic_responses or [])
    editor_model = FakeModelClient(editor_responses or [])
    state_store = store or InMemoryConversationStateStore()
    workflow = WritingWorkflow(
        writer=LlmWriter(writer_model),
        critic=LlmCritic(critic_model),
        editor=LlmEditor(editor_model),
    )
    service = ConversationService(
        coordinator=LlmCoordinator(coordinator_model),
        talker=LlmTalker(talker_model),
        workflow=workflow,
        store=state_store,
        identifier_generator=Ids(),
        clock=lambda: NOW,
        max_recent_messages=50,
    )
    return (
        service,
        state_store,
        coordinator_model,
        talker_model,
        writer_model,
        critic_model,
        editor_model,
    )


def awaiting_store() -> InMemoryConversationStateStore:
    store = InMemoryConversationStateStore()
    critic_report = CriticReport(CriticVerdict.PASS, "Previous review passed.")
    active_task = WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=WritingBrief("Original request", ("Keep it concise.",)),
        status=WritingTaskStatus.AWAITING_USER_EVALUATION,
        created_at=NOW,
        updated_at=NOW,
        working_draft="Existing working draft",
        critic_report=critic_report,
    )
    store.save(
        ConversationState(
            "conversation-1",
            status=ConversationStatus.AWAITING_USER_EVALUATION,
            active_task=active_task,
        )
    )
    return store


def test_chat_route_completes_with_real_model_backed_agents() -> None:
    service, store, coordinator, talker, writer, critic, editor = build_service(
        coordinator_responses=[decision("chat")],
        talker_responses=[response("Hello! How can I help?")],
    )

    messages = service.process_message("conversation-1", "Hello")

    assert [message.content for message in messages] == ["Hello! How can I help?"]
    assert len(coordinator.requests) == len(talker.requests) == 1
    assert writer.requests == critic.requests == editor.requests == []
    assert len(store.load("conversation-1").recent_messages) == 2  # type: ignore[union-attr]


def test_start_writing_pass_completes_with_expected_order() -> None:
    service, store, _, _, writer, critic, editor = build_service(
        coordinator_responses=[
            decision("start_writing_task", task_input="Write a launch post.")
        ],
        writer_responses=[response("Writer output")],
        critic_responses=[report("pass")],
    )

    messages = service.process_message("conversation-1", "Write a launch post")

    assert messages[0].content == "Writer output:\nWriter output"
    assert messages[1].content.startswith("Critic verdict: PASS")
    assert messages[2].content == "The Writer output is now the working draft."
    assert len(writer.requests) == len(critic.requests) == 1
    assert editor.requests == []
    state = store.load("conversation-1")
    assert state is not None
    assert state.active_task is not None
    assert state.active_task.working_draft == "Writer output"
    assert writer.requests[0].structured_output is None
    assert critic.requests[0].structured_output == CRITIC_STRUCTURED_OUTPUT


def test_real_structured_agents_attach_their_provider_neutral_specs() -> None:
    service, _, coordinator, _, _, critic, _ = build_service(
        coordinator_responses=[
            decision("start_writing_task", task_input="Write a launch post.")
        ],
        writer_responses=[response("Writer output")],
        critic_responses=[report("pass")],
    )

    service.process_message("conversation-1", "Write a launch post")

    assert coordinator.requests[0].structured_output == COORDINATOR_STRUCTURED_OUTPUT
    assert critic.requests[0].structured_output == CRITIC_STRUCTURED_OUTPUT


def test_start_writing_revise_completes_optional_editor_pass() -> None:
    service, store, _, _, writer, critic, editor = build_service(
        coordinator_responses=[
            decision("start_writing_task", task_input="Write a launch post.")
        ],
        writer_responses=[response("Writer output")],
        critic_responses=[report("revise")],
        editor_responses=[response("Editor output")],
    )

    messages = service.process_message("conversation-1", "Write a launch post")

    assert messages[0].content == "Writer output:\nWriter output"
    assert messages[1].content.startswith("Critic verdict: REVISE")
    assert messages[2].content == "Revised working draft:\nEditor output"
    assert len(writer.requests) == len(critic.requests) == len(editor.requests) == 1
    state = store.load("conversation-1")
    assert state is not None
    assert state.active_task is not None
    assert state.active_task.working_draft == "Editor output"


def test_approval_completes_without_writing_agents() -> None:
    service, store, _, talker, writer, critic, editor = build_service(
        coordinator_responses=[decision("approve_task")],
        talker_responses=[response("Great, it is approved.")],
        store=awaiting_store(),
    )

    messages = service.process_message("conversation-1", "Looks great")

    assert [message.content for message in messages] == ["Great, it is approved."]
    assert len(talker.requests) == 1
    assert writer.requests == critic.requests == editor.requests == []
    state = store.load("conversation-1")
    assert state is not None
    assert state.status is ConversationStatus.CHATTING
    assert state.active_task is not None
    assert state.active_task.status is WritingTaskStatus.APPROVED


def test_revision_reruns_normal_writing_workflow() -> None:
    service, store, _, _, writer, critic, editor = build_service(
        coordinator_responses=[
            decision("revise_task", revision_instructions="Make it shorter.")
        ],
        writer_responses=[response("Writer output")],
        critic_responses=[report("pass")],
        store=awaiting_store(),
    )

    messages = service.process_message("conversation-1", "Make it shorter")

    assert messages[0].content == "Writer output:\nWriter output"
    assert len(writer.requests) == len(critic.requests) == 1
    assert editor.requests == []
    writer_prompt = writer.requests[0].input
    assert isinstance(writer_prompt, str)
    assert '"current_working_draft": "Existing working draft"' in writer_prompt
    assert '"instructions": ["Keep it concise.", "Make it shorter."]' in writer_prompt
    state = store.load("conversation-1")
    assert state is not None
    assert state.active_task is not None
    assert state.active_task.id == "task-1"
    assert state.active_task.working_draft == "Writer output"
