from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from telegram.constants import ChatType

from editorial_team.agents import AgentError, LlmCoordinator
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingTask,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.interfaces.telegram import GENERIC_TURN_ERROR, TelegramAdapter
from editorial_team.models import FakeModelClient, ModelResponse
from editorial_team.tracing import error_category
from editorial_team.workflows import WritingWorkflow

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


@dataclass
class FixedCoordinator:
    output: object

    def decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output  # type: ignore[return-value]


@dataclass
class FixedTalker:
    output: object

    def respond(self, state: ConversationState, user_message: Message) -> str:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output  # type: ignore[return-value]


@dataclass
class FixedWorkflow:
    output: object

    def execute(self, task: WritingTask) -> EditorialResult:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output  # type: ignore[return-value]


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"id-{self.value}"


class Bot:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[dict[str, object]] = []

    async def send_message(self, **kwargs: object) -> None:
        if self.fail:
            raise RuntimeError("DELIVERY-RAW-SECRET")
        self.sent.append(kwargs)


def update(update_id: int, text: str = "USER-CONTENT-SECRET") -> SimpleNamespace:
    return SimpleNamespace(
        update_id=update_id,
        effective_chat=SimpleNamespace(id=123, type=ChatType.PRIVATE),
        effective_message=SimpleNamespace(text=text),
    )


def service(
    *,
    coordinator: object,
    talker: object = "Safe reply",
    workflow: object = RuntimeError("unused"),
) -> ConversationService:
    return ConversationService(
        coordinator=(
            coordinator
            if hasattr(coordinator, "decide")
            else FixedCoordinator(coordinator)
        ),  # type: ignore[arg-type]
        talker=FixedTalker(talker),
        workflow=(
            workflow if hasattr(workflow, "execute") else FixedWorkflow(workflow)
        ),  # type: ignore[arg-type]
        store=InMemoryConversationStateStore(),
        identifier_generator=Ids(),
        clock=lambda: NOW,
        max_recent_messages=20,
    )


def run_turn(
    application_service: ConversationService,
    *,
    update_id: int = 101,
    bot: Bot | None = None,
) -> Bot:
    telegram_bot = bot or Bot()
    adapter = TelegramAdapter(application_service)
    context = SimpleNamespace(bot=telegram_bot)
    asyncio.run(adapter.handle_text(update(update_id), context))
    return telegram_bot


def events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "editorial_team.live_trace"
    ]


def test_successful_turn_logs_start_route_stages_delivery_and_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker="Safe reply",
    )

    run_turn(app)

    trace = events(caplog)
    assert [line.split()[0] for line in trace] == [
        "telegram_turn_started",
        "coordinator_started",
        "coordinator_completed",
        "route_started",
        "talker_started",
        "talker_completed",
        "assistant_delivery_started",
        "assistant_delivery_completed",
        "telegram_turn_completed",
    ]
    assert "route=chat" in trace[2]
    assert "assistant_message_count=1" in trace[-2]
    assert "chunk_count=1" in trace[-2]
    assert all("correlation_id=tg-101" in line for line in trace)
    assert all("update_id=101" in line for line in trace)


@pytest.mark.parametrize(
    ("model_text", "category"),
    [
        (" ", "blank_response"),
        ("not-json", "json_decoding_failure"),
        (
            json.dumps({"route": "chat", "confidence": 1, "unknown": True}),
            "schema_validation_failure",
        ),
        (
            json.dumps(
                {
                    "route": "start_writing_task",
                    "confidence": 1,
                    "task_input": None,
                }
            ),
            "domain_consistency_failure",
        ),
    ],
)
def test_coordinator_failure_categories_are_sanitized(
    caplog: pytest.LogCaptureFixture,
    model_text: str,
    category: str,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    model = FakeModelClient([ModelResponse(model_text, (), None)])
    app = service(coordinator=LlmCoordinator(model))

    bot = run_turn(app)

    trace = "\n".join(events(caplog))
    assert "coordinator_started" in trace
    assert (
        f"coordinator_failed correlation_id=tg-101 update_id=101 "
        f"stage=coordinator outcome=failed error_category={category}"
    ) in trace
    assert "telegram_turn_failed" in trace
    assert "stage=coordinator" in trace
    assert bot.sent == [{"chat_id": 123, "text": GENERIC_TURN_ERROR}]
    if model_text.strip():
        assert model_text not in trace


def test_provider_failure_is_classified_without_raw_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=FixedCoordinator(
            AgentError("Coordinator model call failed")
        )
    )

    run_turn(app)

    trace = "\n".join(events(caplog))
    assert "error_category=provider_model_failure" in trace
    assert "Coordinator model call failed" not in trace


def test_talker_failure_is_distinct_after_successful_coordinator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker=AgentError("Talker model call failed"),
    )

    run_turn(app)

    trace = "\n".join(events(caplog))
    assert "coordinator_completed" in trace
    assert "route=chat" in trace
    assert "talker_failed" in trace
    assert "telegram_turn_failed" in trace
    assert "stage=talker" in trace
    assert "Talker model call failed" not in trace


def test_writing_failure_is_distinct_after_successful_coordinator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="DRAFT-CONTENT-SECRET",
    )
    app = service(
        coordinator=decision,
        workflow=AgentError("Writer model call failed"),
    )

    run_turn(app)

    trace = "\n".join(events(caplog))
    assert "coordinator_completed" in trace
    assert "route=start_writing_task" in trace
    assert "writing_workflow_started" in trace
    assert "writing_workflow_failed" in trace
    assert "telegram_turn_failed" in trace
    assert "stage=writing_workflow" in trace
    assert "Writer model call failed" not in trace
    assert "DRAFT-CONTENT-SECRET" not in trace


def test_successful_writing_logs_granular_stages_and_safe_result_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    report = CriticReport(
        CriticVerdict.REVISE,
        "CRITIC-SUMMARY-SECRET",
        (
            CriticIssue(
                CriticIssueSeverity.MAJOR,
                "CRITIC-ISSUE-SECRET",
            ),
        ),
    )

    class Writer:
        def write(self, task: WritingTask) -> str:
            return "WRITER-OUTPUT-SECRET"

    class Critic:
        def review(self, task: WritingTask, draft: str) -> CriticReport:
            return report

    class Editor:
        def revise(
            self,
            task: WritingTask,
            draft: str,
            critic_report: CriticReport,
        ) -> str:
            return "EDITOR-OUTPUT-SECRET"

    workflow = WritingWorkflow(writer=Writer(), critic=Critic(), editor=Editor())
    decision = CoordinatorDecision(
        CoordinatorRoute.START_WRITING_TASK,
        1.0,
        task_input="TASK-INPUT-SECRET",
    )
    app = service(coordinator=decision, workflow=workflow)

    run_turn(app)

    trace = "\n".join(events(caplog))
    for event in (
        "writing_workflow_started",
        "writer_started",
        "writer_completed",
        "critic_started",
        "critic_completed",
        "editor_started",
        "editor_completed",
        "writing_workflow_completed",
    ):
        assert event in trace
    assert "critic_verdict=revise" in trace
    assert "revision_applied=true" in trace
    for forbidden in (
        "TASK-INPUT-SECRET",
        "WRITER-OUTPUT-SECRET",
        "CRITIC-SUMMARY-SECRET",
        "CRITIC-ISSUE-SECRET",
        "EDITOR-OUTPUT-SECRET",
    ):
        assert forbidden not in trace


def test_delivery_failure_is_distinct_and_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker="MODEL-RESPONSE-SECRET",
    )

    with pytest.raises(RuntimeError, match="DELIVERY-RAW-SECRET"):
        run_turn(app, bot=Bot(fail=True))

    trace = "\n".join(events(caplog))
    assert "assistant_delivery_started" in trace
    assert "telegram_turn_failed" in trace
    assert "stage=assistant_delivery" in trace
    assert "error_category=runtime_error" in trace
    assert "DELIVERY-RAW-SECRET" not in trace
    assert "MODEL-RESPONSE-SECRET" not in trace


def test_logs_exclude_content_prompts_and_secret_markers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker="DRAFT-AND-MODEL-OUTPUT-SECRET",
    )

    run_turn(app)

    trace = "\n".join(events(caplog))
    for forbidden in (
        "USER-CONTENT-SECRET",
        "DRAFT-AND-MODEL-OUTPUT-SECRET",
        "APPLICATION INSTRUCTIONS",
        "TELEGRAM-TOKEN-SECRET",
        "GEMINI-KEY-SECRET",
    ):
        assert forbidden not in trace


def test_separate_turns_have_distinct_consistent_correlations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    app = service(
        coordinator=CoordinatorDecision(CoordinatorRoute.CHAT, 1.0),
        talker="Safe reply",
    )
    adapter = TelegramAdapter(app)
    context = SimpleNamespace(bot=Bot())

    asyncio.run(adapter.handle_text(update(201), context))
    asyncio.run(adapter.handle_text(update(202), context))

    trace = events(caplog)
    first = [line for line in trace if "update_id=201" in line]
    second = [line for line in trace if "update_id=202" in line]
    assert first and second
    assert all("correlation_id=tg-201" in line for line in first)
    assert all("correlation_id=tg-202" in line for line in second)
    assert not any("correlation_id=tg-202" in line for line in first)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (AgentError("Coordinator model call failed"), "provider_model_failure"),
        (AgentError("Coordinator returned invalid output"), "blank_response"),
        (AgentError("Model returned invalid JSON"), "json_decoding_failure"),
        (
            AgentError("Model returned unexpected structured-output fields"),
            "schema_validation_failure",
        ),
        (
            AgentError("Coordinator returned invalid structured output"),
            "domain_consistency_failure",
        ),
    ],
)
def test_error_category_never_contains_raw_error_text(
    error: AgentError,
    category: str,
) -> None:
    assert error_category(error) == category
    assert str(error) not in category
