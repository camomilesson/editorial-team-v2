"""Behavior tests using the real compiled durable production graph."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from editorial_team.artifacts import ParagraphChunker, SQLiteArtifactStore
from editorial_team.conversation import ConversationService, ConversationServiceError
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    WritingTask,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.graphs import build_parent_graph, create_sqlite_checkpointer
from editorial_team.runtime import RuntimeJobSource, RuntimeQueue, RuntimeQueueError
from editorial_team.tracing import bind_turn_trace, trace_for_update

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"id-{self.value}"


@dataclass
class Coordinator:
    seen: list[ConversationState] = field(default_factory=list)

    def decide(self, state: ConversationState, message: Message) -> CoordinatorDecision:
        self.seen.append(state)
        if message.content.startswith("write "):
            return CoordinatorDecision(
                CoordinatorRoute.START_WRITING_TASK,
                1.0,
                task_input=message.content.removeprefix("write "),
            )
        if message.content.startswith("revise "):
            return CoordinatorDecision(
                CoordinatorRoute.REVISE_TASK,
                1.0,
                revision_instructions=message.content.removeprefix("revise "),
            )
        return CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)


@dataclass
class Talker:
    seen: list[ConversationState] = field(default_factory=list)

    def respond(self, state: ConversationState, message: Message) -> str:
        self.seen.append(state)
        return f"Chat: {message.content}"


@dataclass
class Writer:
    calls: list[WritingTask] = field(default_factory=list)

    def write(self, task: WritingTask) -> str:
        self.calls.append(task)
        instruction = task.brief.instructions[-1] if task.brief.instructions else "initial"
        return f"{task.brief.original_request}|{instruction}"


@dataclass
class Critic:
    verdict: CriticVerdict = CriticVerdict.PASS
    calls: int = 0

    def review(self, task: WritingTask, draft: str) -> CriticReport:
        self.calls += 1
        if self.verdict is CriticVerdict.PASS:
            return CriticReport(CriticVerdict.PASS, "Approved.")
        return CriticReport(
            CriticVerdict.REVISE,
            "Revise.",
            (CriticIssue(CriticIssueSeverity.MAJOR, "Improve it."),),
        )


@dataclass
class Editor:
    calls: int = 0

    def revise(self, task: WritingTask, draft: str, report: CriticReport) -> str:
        self.calls += 1
        return f"edited:{draft}"


@dataclass
class Actors:
    coordinator: Coordinator = field(default_factory=Coordinator)
    talker: Talker = field(default_factory=Talker)
    writer: Writer = field(default_factory=Writer)
    critic: Critic = field(default_factory=Critic)
    editor: Editor = field(default_factory=Editor)


@dataclass
class ComposedService:
    service: ConversationService
    runner: object
    artifact_store: SQLiteArtifactStore

    def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
        return self.service.process_message(conversation_id, text)

    def close(self) -> None:
        self.service.close()


def service_for(
    database: Path,
    actors: Actors,
    *,
    busy_timeout_seconds: float = 5.0,
    identifier_generator: Callable[[], str] | None = None,
) -> ComposedService:
    saver, close = create_sqlite_checkpointer(database, busy_timeout_seconds=busy_timeout_seconds)
    artifact_store = SQLiteArtifactStore(
        database.with_name(f"{database.stem}-artifacts.db"),
        chunker=ParagraphChunker(),
    )
    artifact_store.initialize()
    graph = build_parent_graph(
        coordinator=actors.coordinator,
        talker=actors.talker,
        writer=actors.writer,
        critic=actors.critic,
        editor=actors.editor,
        identifier_generator=identifier_generator or (lambda: uuid4().hex),
        clock=lambda: NOW,
        max_recent_messages=50,
        artifact_store=artifact_store,
    ).compile(checkpointer=saver)

    def close_all() -> None:
        close()
        artifact_store.close()

    return ComposedService(
        ConversationService(graph_runner=graph, close_checkpointer=close_all),
        graph,
        artifact_store,
    )


def conversation(service: ComposedService, conversation_id: str) -> ConversationState:
    snapshot = service.runner.get_state(  # type: ignore[attr-defined]
        {"configurable": {"thread_id": f"editorial:v1:{conversation_id}"}}
    )
    value = snapshot.values["conversation"]
    assert isinstance(value, ConversationState)
    return value


def test_same_thread_continues_chat_task_revision_and_replacement(tmp_path: Path) -> None:
    actors = Actors()
    service = service_for(tmp_path / "state.db", actors)
    service.process_message("telegram-chat-1", "hello")
    assert actors.coordinator.seen[-1].recent_messages[-1].content == "hello"
    service.process_message("telegram-chat-1", "write first post")
    first = conversation(service, "telegram-chat-1")
    assert first.active_task is not None
    first_id = first.active_task.id
    service.process_message("telegram-chat-1", "thanks")
    assert conversation(service, "telegram-chat-1").active_task.id == first_id  # type: ignore[union-attr]
    service.process_message("telegram-chat-1", "revise shorter")
    revised = conversation(service, "telegram-chat-1")
    assert revised.active_task is not None
    assert revised.active_task.id != first_id
    assert revised.active_task.brief.instructions == ("shorter",)
    assert actors.writer.calls[-1].working_draft == "first post|initial"
    service.process_message("telegram-chat-1", "write second post")
    replacement = conversation(service, "telegram-chat-1")
    assert replacement.active_task is not None
    assert replacement.active_task.id != first_id
    service.close()


def test_threads_are_isolated(tmp_path: Path) -> None:
    service = service_for(tmp_path / "state.db", Actors())
    for conversation_id, brief in (
        ("telegram-chat-1", "one"),
        ("telegram-chat-2", "two"),
        ("telegram-chat-n100", "group-a"),
        ("telegram-chat-n200", "group-b"),
        ("telegram-chat-n100-thread-7", "topic-a"),
        ("telegram-chat-n100-thread-8", "topic-b"),
    ):
        service.process_message(conversation_id, f"write {brief}")
    drafts = {
        conversation(service, key).active_task.working_draft  # type: ignore[union-attr]
        for key in (
            "telegram-chat-1",
            "telegram-chat-2",
            "telegram-chat-n100",
            "telegram-chat-n200",
            "telegram-chat-n100-thread-7",
            "telegram-chat-n100-thread-8",
        )
    }
    assert len(drafts) == 6
    service.close()


def test_restart_recovers_task_and_revision_context(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    first_actors = Actors()
    first = service_for(database, first_actors)
    first.process_message("telegram-chat-1", "write persistent draft")
    task_id = conversation(first, "telegram-chat-1").active_task.id  # type: ignore[union-attr]
    first.close()

    second_actors = Actors()
    second = service_for(database, second_actors)
    second.process_message("telegram-chat-1", "revise warmer")
    restored = conversation(second, "telegram-chat-1")
    assert restored.active_task is not None
    assert restored.active_task.id != task_id
    assert restored.active_task.brief.instructions == ("warmer",)
    assert second_actors.writer.calls[0].working_draft == "persistent draft|initial"
    second.close()


def test_critic_controls_editor_execution(tmp_path: Path) -> None:
    passing = Actors()
    service = service_for(tmp_path / "pass.db", passing)
    service.process_message("telegram-chat-1", "write pass")
    assert passing.critic.calls == 1
    assert passing.editor.calls == 0
    artifacts = service.artifact_store.list_artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].producer.value == "writer"
    assert artifacts[0].user_request == "pass"
    service.close()


def test_completed_editorial_runs_store_exact_participants_and_new_run_ids(
    tmp_path: Path,
) -> None:
    actors = Actors()
    actors.critic.verdict = CriticVerdict.REVISE
    service = service_for(tmp_path / "artifacts.db", actors)
    service.process_message("telegram-chat-1", "hello")
    assert service.artifact_store.list_artifacts() == ()

    service.process_message("telegram-chat-1", "write launch note")
    first = service.artifact_store.list_artifacts()
    assert {artifact.producer.value for artifact in first} == {"editor", "writer"}
    assert len({artifact.task_id for artifact in first}) == 1
    assert {artifact.user_request for artifact in first} == {"launch note"}

    service.process_message("telegram-chat-1", "revise shorter")
    all_artifacts = service.artifact_store.list_artifacts()
    assert len(all_artifacts) == 4
    run_ids = {artifact.task_id for artifact in all_artifacts}
    assert len(run_ids) == 2
    revision = [artifact for artifact in all_artifacts if artifact.user_request == "shorter"]
    assert {artifact.producer.value for artifact in revision} == {"editor", "writer"}
    assert len({artifact.task_id for artifact in revision}) == 1
    service.close()


def test_artifact_persistence_failure_does_not_finalize_turn(tmp_path: Path) -> None:
    actors = Actors()
    service = service_for(tmp_path / "persistence-failure.db", actors)
    original = service.artifact_store.save_run

    def fail(artifacts: object) -> None:
        del artifacts
        raise RuntimeError("private database detail")

    service.artifact_store.save_run = fail  # type: ignore[method-assign]
    with pytest.raises(ConversationServiceError, match="artifacts could not be saved"):
        service.process_message("telegram-chat-1", "write unsaved")
    assert service.artifact_store.list_artifacts() == ()
    service.artifact_store.save_run = original  # type: ignore[method-assign]
    service.process_message("telegram-chat-1", "write recovered")
    assert len(service.artifact_store.list_artifacts()) == 1
    service.close()


def test_final_turn_assembly_failure_saves_no_artifacts(tmp_path: Path) -> None:
    identifiers = Ids()

    def fail_during_assistant_messages() -> str:
        if identifiers.value == 3:
            raise RuntimeError("identifier unavailable")
        return identifiers()

    service = service_for(
        tmp_path / "finalization-failure.db",
        Actors(),
        identifier_generator=fail_during_assistant_messages,
    )
    with pytest.raises(ConversationServiceError):
        service.process_message("telegram-chat-1", "write unsaved")
    assert service.artifact_store.list_artifacts() == ()
    service.close()


def test_locked_database_fails_within_bound_and_shared_worker_recovers(tmp_path: Path) -> None:
    async def scenario() -> tuple[float, list[str]]:
        database = tmp_path / "locked.db"
        service = service_for(database, Actors(), busy_timeout_seconds=0.05)
        queue = RuntimeQueue()
        await queue.start()
        locker = sqlite3.connect(database)
        locker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeQueueError, match="Runtime job failed"):
                await queue.submit(
                    source=RuntimeJobSource.TELEGRAM,
                    correlation_id="locked-turn",
                    operation=lambda: asyncio.to_thread(
                        service.process_message, "telegram-chat-1", "hello"
                    ),
                )
        finally:
            elapsed = time.monotonic() - started
            locker.rollback()
            locker.close()
        completed: list[str] = []
        await queue.submit(
            source=RuntimeJobSource.HEARTBEAT,
            correlation_id="heartbeat-after-lock",
            operation=lambda: _record(completed, "heartbeat"),
        )
        await queue.submit(
            source=RuntimeJobSource.TELEGRAM,
            correlation_id="turn-after-lock",
            operation=lambda: asyncio.to_thread(
                service.process_message, "telegram-chat-1", "hello again"
            ),
        )
        completed.append("telegram")
        await queue.close()
        service.close()
        return elapsed, completed

    async def _record(items: list[str], value: str) -> None:
        items.append(value)

    elapsed, completed = asyncio.run(scenario())
    assert elapsed < 1.0
    assert completed == ["heartbeat", "telegram"]


@pytest.mark.parametrize("participant", ["writer", "critic"])
@pytest.mark.parametrize("malformed", [False, True])
def test_failed_new_task_preserves_previous_active_task_and_later_turn_recovers(
    tmp_path: Path, participant: str, malformed: bool
) -> None:
    actors = Actors()
    service = service_for(tmp_path / f"{participant}-{malformed}.db", actors)
    service.process_message("telegram-chat-1", "write canonical")
    artifact_count = len(service.artifact_store.list_artifacts())
    original = getattr(actors, participant)
    method_name = "write" if participant == "writer" else "review"
    original_method = getattr(original, method_name)

    def fail(*args: object) -> object:
        del args
        if malformed:
            return " " if participant == "writer" else object()
        raise RuntimeError("private provider diagnostics")

    setattr(original, method_name, fail)
    with pytest.raises(ConversationServiceError):
        service.process_message("telegram-chat-1", "write replacement")
    assert len(service.artifact_store.list_artifacts()) == artifact_count
    setattr(original, method_name, original_method)

    service.process_message("telegram-chat-1", "recovered chat")
    recovered = actors.coordinator.seen[-1]
    assert recovered.active_task is not None
    assert recovered.active_task.working_draft == "canonical|initial"
    service.close()


@pytest.mark.parametrize("malformed", [False, True])
def test_failed_editor_preserves_canonical_draft_and_later_turn_recovers(
    tmp_path: Path, malformed: bool
) -> None:
    actors = Actors()
    service = service_for(tmp_path / f"editor-{malformed}.db", actors)
    service.process_message("telegram-chat-1", "write canonical")
    artifact_count = len(service.artifact_store.list_artifacts())
    actors.critic.verdict = CriticVerdict.REVISE
    original = actors.editor.revise

    def fail(*args: object) -> object:
        del args
        if malformed:
            return " "
        raise RuntimeError("private editor diagnostics")

    actors.editor.revise = fail  # type: ignore[method-assign]
    with pytest.raises(ConversationServiceError):
        service.process_message("telegram-chat-1", "revise shorter")
    assert len(service.artifact_store.list_artifacts()) == artifact_count
    actors.editor.revise = original  # type: ignore[method-assign]

    service.process_message("telegram-chat-1", "recovered chat")
    recovered = actors.coordinator.seen[-1]
    assert recovered.active_task is not None
    assert recovered.active_task.working_draft == "canonical|initial"
    service.close()


@pytest.mark.parametrize("participant", ["coordinator", "talker"])
def test_failed_conversation_node_is_atomic_and_later_turn_recovers(
    tmp_path: Path, participant: str
) -> None:
    actors = Actors()
    service = service_for(tmp_path / f"{participant}.db", actors)
    service.process_message("telegram-chat-1", "before failure")
    target = getattr(actors, participant)
    method_name = "decide" if participant == "coordinator" else "respond"
    original = getattr(target, method_name)

    def fail(*args: object) -> object:
        del args
        raise RuntimeError("private diagnostics")

    setattr(target, method_name, fail)
    with pytest.raises(ConversationServiceError):
        service.process_message("telegram-chat-1", "failed content")
    setattr(target, method_name, original)

    service.process_message("telegram-chat-1", "after failure")
    recovered = actors.coordinator.seen[-1]
    contents = [message.content for message in recovered.recent_messages]
    assert "before failure" in contents
    assert "failed content" not in contents
    assert "after failure" in contents
    service.close()


def test_graph_tracing_has_stage_order_and_excludes_product_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="editorial_team.live_trace")
    service = service_for(tmp_path / "trace.db", Actors())
    secret = "SYNTHETIC_PRIVATE_DRAFT"

    with bind_turn_trace(trace_for_update(41)):
        service.process_message("telegram-chat-1", f"write {secret}")

    events = [record.message.split()[0] for record in caplog.records]
    assert events.index("coordinator_started") < events.index("writer_started")
    assert events.index("writer_started") < events.index("critic_started")
    assert "writing_workflow_completed" in events
    assert all("correlation_id=tg-41" in record.message for record in caplog.records)
    assert secret not in caplog.text
    service.close()


def test_separate_graph_turns_have_distinct_trace_correlations(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="editorial_team.live_trace")
    service = service_for(tmp_path / "trace-separate.db", Actors())
    with bind_turn_trace(trace_for_update(51)):
        service.process_message("telegram-chat-1", "hello")
    with bind_turn_trace(trace_for_update(52)):
        service.process_message("telegram-chat-1", "again")

    assert "correlation_id=tg-51" in caplog.text
    assert "correlation_id=tg-52" in caplog.text
    service.close()

    revising = Actors(critic=Critic(CriticVerdict.REVISE))
    service = service_for(tmp_path / "revise.db", revising)
    service.process_message("telegram-chat-1", "write revise")
    assert revising.critic.calls == 1
    assert revising.editor.calls == 1
    assert conversation(service, "telegram-chat-1").active_task.working_draft.startswith("edited:")  # type: ignore[union-attr]
    service.close()
