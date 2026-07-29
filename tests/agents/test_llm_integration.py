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
from editorial_team.domain.conversation import ConversationState
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
        status=WritingTaskStatus.REVIEWED,
        created_at=NOW,
        updated_at=NOW,
        working_draft="Existing working draft",
        critic_report=critic_report,
    )
    store.save(
        ConversationState(
            "conversation-1",
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

    assert [message.content for message in messages] == [
        "Talker\n\nHello! How can I help?"
    ]
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

    assert [message.content for message in messages] == [
        "Writer\n\nWriter output",
        "Critic\n\nVerdict: PASS\n\nSummary: Review complete.\n\nIssues: None",
        "Editor\n\nWorking draft approved, see above.",
    ]
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

    assert messages[0].content == "Writer\n\nWriter output"
    assert messages[1].content.startswith("Critic\n\nVerdict: REVISE")
    assert messages[2].content == "Editor\n\nEditor output"
    assert len(writer.requests) == len(critic.requests) == len(editor.requests) == 1
    state = store.load("conversation-1")
    assert state is not None
    assert state.active_task is not None
    assert state.active_task.working_draft == "Editor output"


def test_praise_routes_to_talker_and_preserves_latest_task() -> None:
    service, store, _, talker, writer, critic, editor = build_service(
        coordinator_responses=[decision("chat")],
        talker_responses=[response("Thanks! Let me know what you want to work on next.")],
        store=awaiting_store(),
    )

    messages = service.process_message("conversation-1", "Looks great")

    assert [message.content for message in messages] == [
        "Talker\n\nThanks! Let me know what you want to work on next."
    ]
    assert len(talker.requests) == 1
    assert writer.requests == critic.requests == editor.requests == []
    state = store.load("conversation-1")
    assert state is not None
    assert state.active_task is not None
    assert state.active_task.status is WritingTaskStatus.REVIEWED


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

    assert messages[0].content == "Writer\n\nWriter output"
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


def test_latest_task_survives_chat_then_revision_and_is_replaced_by_new_task() -> None:
    service, store, _, talker, writer, critic, editor = build_service(
        coordinator_responses=[
            decision("start_writing_task", task_input="Write the first post."),
            decision("chat"),
            decision("chat"),
            decision("revise_task", revision_instructions="Make it shorter."),
            decision("start_writing_task", task_input="Write a new announcement."),
        ],
        talker_responses=[
            response("Thanks!"),
            response("What would you like to change?"),
        ],
        writer_responses=[
            response("First draft"),
            response("Shorter draft"),
            response("New task draft"),
        ],
        critic_responses=[report("pass"), report("pass"), report("pass")],
    )

    first_messages = service.process_message("conversation-1", "Write the first post")
    first_state = store.load("conversation-1")
    assert first_state is not None and first_state.active_task is not None
    first_task_id = first_state.active_task.id
    assert len(first_messages) == 3

    praise = service.process_message("conversation-1", "Awesome")
    uncertain = service.process_message("conversation-1", "I’m not sure about it")
    after_chat = store.load("conversation-1")
    assert [message.content for message in (*praise, *uncertain)] == [
        "Talker\n\nThanks!",
        "Talker\n\nWhat would you like to change?",
    ]
    assert after_chat is not None and after_chat.active_task is not None
    assert after_chat.active_task.id == first_task_id

    revision = service.process_message("conversation-1", "Actually, make it shorter")
    revised_state = store.load("conversation-1")
    assert len(revision) == 3
    assert revised_state is not None and revised_state.active_task is not None
    assert revised_state.active_task.id == first_task_id
    assert revised_state.active_task.brief.instructions == ("Make it shorter.",)
    assert revised_state.active_task.working_draft == "Shorter draft"

    new_task = service.process_message("conversation-1", "Write a new announcement")
    final_state = store.load("conversation-1")
    assert len(new_task) == 3
    assert final_state is not None and final_state.active_task is not None
    assert final_state.active_task.id != first_task_id
    assert final_state.active_task.brief.original_request == "Write a new announcement."
    assert len(talker.requests) == 2
    assert len(writer.requests) == len(critic.requests) == 3
    assert editor.requests == []
