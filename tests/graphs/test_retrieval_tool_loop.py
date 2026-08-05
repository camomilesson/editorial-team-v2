"""Focused integration tests for the Coordinator-owned LangChain tool loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from editorial_team.agents import ToolCallingCoordinator
from editorial_team.artifacts.models import ArtifactProducer
from editorial_team.artifacts.retrieval_types import RetrievedDraft, SearchResult
from editorial_team.conversation import ConversationService, ConversationServiceError
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    WritingTask,
)
from editorial_team.graphs import build_parent_graph, create_sqlite_checkpointer

NOW = datetime(2026, 3, 29, 0, 30, tzinfo=UTC)


class LegacyCoordinator:
    def decide(self, state: object, message: object) -> object:
        del state, message
        raise AssertionError("the retrieval-aware coordinator must own this turn")


@dataclass
class ScriptedChatModel:
    responses: list[AIMessage]
    bindings: list[tuple[Any, ...]] = field(default_factory=list)
    invocations: list[list[Any]] = field(default_factory=list)

    def bind_tools(self, tools: list[Any]) -> ScriptedChatModel:
        self.bindings.append(tuple(tools))
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(messages)
        return self.responses.pop(0)


@dataclass
class Retriever:
    draft: RetrievedDraft | None = None
    results: tuple[SearchResult, ...] = ()
    searches: list[dict[str, Any]] = field(default_factory=list)
    gets: list[tuple[str, str]] = field(default_factory=list)

    def search(self, **kwargs: Any) -> tuple[SearchResult, ...]:
        self.searches.append(kwargs)
        return self.results

    def get_draft(self, *, artifact_id: str, conversation_id: str) -> RetrievedDraft | None:
        self.gets.append((artifact_id, conversation_id))
        if self.draft is None or self.draft.conversation_id != conversation_id:
            return None
        return self.draft if self.draft.artifact_id == artifact_id else None


@dataclass
class Talker:
    contexts: list[Any] = field(default_factory=list)

    def respond(self, state: object, message: object, context: object = None) -> str:
        del state, message
        self.contexts.append(context)
        return "Could you clarify which draft you mean?"


@dataclass
class Writer:
    tasks: list[WritingTask] = field(default_factory=list)

    def write(self, task: WritingTask) -> str:
        self.tasks.append(task)
        return f"rewritten:{task.working_draft}"


@dataclass
class Critic:
    verdict: CriticVerdict = CriticVerdict.PASS

    def review(self, task: WritingTask, draft: str) -> CriticReport:
        del task, draft
        if self.verdict is CriticVerdict.PASS:
            return CriticReport(CriticVerdict.PASS, "Approved")
        return CriticReport(
            CriticVerdict.REVISE,
            "Revise it",
            (CriticIssue(CriticIssueSeverity.MAJOR, "Improve it"),),
        )


class Editor:
    def revise(self, task: WritingTask, draft: str, report: CriticReport) -> str:
        del task, draft, report
        return "edited historical output"


@dataclass
class Store:
    runs: list[tuple[Any, ...]] = field(default_factory=list)

    def save_run(self, artifacts: tuple[Any, ...]) -> None:
        self.runs.append(artifacts)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"generated-{self.value}"


def call(name: str, args: dict[str, Any], number: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call-{number}"}],
    )


def final(route: str, **values: Any) -> AIMessage:
    payload = {
        "route": route,
        "confidence": 1.0,
        "task_input": None,
        "revision_instructions": None,
        "talker_context": None,
        **values,
    }
    return AIMessage(content=json.dumps(payload))


def block_final(route: str, **values: Any) -> AIMessage:
    payload = final(route, **values).content
    return AIMessage(
        content=[
            {
                "type": "text",
                "text": payload,
                "extras": {"signature": "gemini-signature"},
            }
        ]
    )


def result(rank: int, artifact_id: str, excerpt: str) -> SearchResult:
    return SearchResult(
        rank,
        f"chunk-{rank}",
        artifact_id,
        f"task-{rank}",
        excerpt,
        NOW,
        ArtifactProducer.WRITER,
        rank,
        0.8,
        rank,
        1.0,
        0.03,
        0.9,
        0,
    )


def service_for(
    model: ScriptedChatModel,
    retriever: Retriever,
    *,
    critic_verdict: CriticVerdict = CriticVerdict.PASS,
    checkpointer: Any = None,
    clock: Any = lambda: NOW,
) -> tuple[ConversationService, Writer, Talker, Store, Any]:
    writer, talker, store = Writer(), Talker(), Store()
    graph = build_parent_graph(
        coordinator=LegacyCoordinator(),
        tool_coordinator=ToolCallingCoordinator(model),
        retriever=retriever,  # type: ignore[arg-type]
        talker=talker,
        writer=writer,
        critic=Critic(critic_verdict),
        editor=Editor(),
        identifier_generator=Ids(),
        clock=clock,
        max_recent_messages=20,
        artifact_store=store,  # type: ignore[arg-type]
    ).compile(checkpointer=checkpointer)
    return ConversationService(graph_runner=graph), writer, talker, store, graph


def active_task(graph: Any, conversation_id: str = "chat-a") -> WritingTask | None:
    snapshot = graph.get_state(
        {"configurable": {"thread_id": f"editorial:v1:{conversation_id}"}}
    )
    return snapshot.values["conversation"].active_task


def test_search_then_explicit_get_draft_starts_new_immutable_run() -> None:
    original = RetrievedDraft(
        artifact_id="artifact-old",
        task_id="task-old",
        producer=ArtifactProducer.WRITER,
        created_at=NOW,
        conversation_id="chat-a",
        user_request="Original request",
        content="Complete historical draft",
    )
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "Aurora"}, 1),
            call("get_draft", {"artifact_id": "artifact-old"}, 2),
            final("start_writing_task", task_input="Make the historical draft shorter"),
        ]
    )
    service, writer, _, store, _ = service_for(model, Retriever(original))

    service.process_message("chat-a", "Make our Aurora draft shorter")

    assert writer.tasks[0].working_draft == original.content
    assert writer.tasks[0].brief.original_request == "Make the historical draft shorter"
    assert writer.tasks[0].id != original.task_id
    assert store.runs[0][0].content == "rewritten:Complete historical draft"
    assert original.content == "Complete historical draft"
    final_invocation = model.invocations[2]
    assert isinstance(final_invocation[-1], ToolMessage)
    assert "Complete historical draft" not in final_invocation[-1].content
    assert '"complete_draft_loaded": true' in final_invocation[-1].content
    assert [tuple(tool.name for tool in binding) for binding in model.bindings] == [
        ("search_corpus", "get_draft")
    ] * 3


@pytest.mark.parametrize(
    "user_text",
    [
        "remember how we worked on the Aurora post? can you pull up the latest draft "
        "and add more emoji to it?",
        "remember the aurora post and amend it to have more emoji",
    ],
)
def test_named_historical_aurora_request_overrides_active_skyrim_task(
    user_text: str,
    tmp_path: Path,
) -> None:
    aurora = RetrievedDraft(
        "aurora-latest",
        "aurora-task",
        ArtifactProducer.EDITOR,
        NOW,
        "chat-a",
        "Make Aurora formal",
        "Complete Aurora draft",
    )
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write a formal Skyrim dragons post"),
            call("search_corpus", {"query": "Aurora post"}, 1),
            call("get_draft", {"artifact_id": "aurora-latest"}, 2),
            final("start_writing_task", task_input="Add more emojis to Aurora"),
        ]
    )
    retriever = Retriever(aurora, (result(1, "aurora-latest", "Aurora excerpt"),))
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, _, store, graph = service_for(
        model, retriever, checkpointer=checkpointer
    )

    try:
        service.process_message("chat-a", "Write a formal Skyrim dragons post")
        skyrim = active_task(graph)
        service.process_message("chat-a", user_text)

        assert retriever.searches[0]["query"] == "Aurora post"
        assert retriever.gets == [("aurora-latest", "chat-a")]
        assert writer.tasks[1].working_draft == "Complete Aurora draft"
        assert writer.tasks[1].working_draft != skyrim.working_draft  # type: ignore[union-attr]
        assert writer.tasks[1].brief.original_request == "Add more emojis to Aurora"
        assert active_task(graph).id != skyrim.id  # type: ignore[union-attr]
        assert len(store.runs) == 2
    finally:
        close()


def test_latest_named_historical_request_prefers_recent_and_loads_complete_draft(
    tmp_path: Path,
) -> None:
    aurora = RetrievedDraft(
        "aurora-latest",
        "aurora-task",
        ArtifactProducer.EDITOR,
        NOW,
        "chat-a",
        "Latest Aurora",
        "Complete latest Aurora artifact",
    )
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write Skyrim"),
            call(
                "search_corpus",
                {"query": "Aurora", "prefer_recent": True},
                1,
            ),
            call("get_draft", {"artifact_id": "aurora-latest"}, 2),
            final("start_writing_task", task_input="Make latest Aurora shorter"),
        ]
    )
    retriever = Retriever(aurora, (result(1, "aurora-latest", "excerpt only"),))
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, _, _, _ = service_for(model, retriever, checkpointer=checkpointer)

    try:
        service.process_message("chat-a", "Write Skyrim")
        service.process_message(
            "chat-a", "Pull up the latest Aurora draft and add more emojis"
        )

        assert retriever.searches[0]["prefer_recent"] is True
        assert retriever.gets == [("aurora-latest", "chat-a")]
        assert writer.tasks[1].working_draft == "Complete latest Aurora artifact"
        assert writer.tasks[1].working_draft != "excerpt only"
    finally:
        close()


@pytest.mark.parametrize("instruction", ["Add more emojis", "Make it more formal"])
def test_unqualified_revision_continues_active_skyrim_without_retrieval(
    instruction: str,
    tmp_path: Path,
) -> None:
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write a formal Skyrim dragons post"),
            final("revise_task", revision_instructions=instruction),
        ]
    )
    retriever = Retriever()
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, _, _, _ = service_for(model, retriever, checkpointer=checkpointer)

    try:
        service.process_message("chat-a", "Write a formal Skyrim dragons post")
        skyrim_draft = writer.tasks[0]
        service.process_message("chat-a", instruction)

        assert retriever.searches == [] and retriever.gets == []
        assert writer.tasks[1].working_draft == f"rewritten:{skyrim_draft.working_draft}"
        assert writer.tasks[1].brief.instructions == (instruction,)
    finally:
        close()


@pytest.mark.parametrize(
    ("results", "reason"),
    [
        ((), "no_match"),
        (
            (
                result(1, "aurora-one", "first Aurora"),
                result(2, "aurora-two", "second Aurora"),
            ),
            "ambiguous_candidates",
        ),
    ],
)
def test_unresolved_historical_search_never_falls_back_to_active_skyrim(
    results: tuple[SearchResult, ...],
    reason: str,
    tmp_path: Path,
) -> None:
    context = {
        "reason": reason,
        "candidate_summaries": [item.excerpt for item in results],
        "recommended_question": "Which Aurora draft should I use?",
    }
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write Skyrim"),
            call("search_corpus", {"query": "Aurora"}, 1),
            final("chat", talker_context=context),
        ]
    )
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, talker, store, graph = service_for(
        model, Retriever(results=results), checkpointer=checkpointer
    )

    try:
        service.process_message("chat-a", "Write Skyrim")
        skyrim = active_task(graph)
        service.process_message("chat-a", "Remember the Aurora post and amend it")

        assert active_task(graph) == skyrim
        assert len(writer.tasks) == 1 and len(store.runs) == 1
        assert talker.contexts[-1].reason.value == reason
    finally:
        close()


def test_failed_historical_get_never_runs_writer_or_revises_active_skyrim(
    tmp_path: Path,
) -> None:
    context = {
        "reason": "tool_problem",
        "candidate_summaries": [],
        "recommended_question": "Could you identify the Aurora draft another way?",
    }
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write Skyrim"),
            call("search_corpus", {"query": "Aurora"}, 1),
            call("get_draft", {"artifact_id": "missing-aurora"}, 2),
            final("chat", talker_context=context),
        ]
    )
    retriever = Retriever(results=(result(1, "missing-aurora", "Aurora"),))
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, _, _, graph = service_for(
        model, retriever, checkpointer=checkpointer
    )

    try:
        service.process_message("chat-a", "Write Skyrim")
        skyrim = active_task(graph)
        service.process_message("chat-a", "Remember Aurora and add emojis")

        assert retriever.gets == [("missing-aurora", "chat-a")]
        assert len(writer.tasks) == 1
        assert active_task(graph) == skyrim
    finally:
        close()


def test_historical_tool_turn_cannot_fall_back_to_active_task_revision(
    tmp_path: Path,
) -> None:
    model = ScriptedChatModel(
        [
            final("start_writing_task", task_input="Write Skyrim"),
            call("search_corpus", {"query": "Aurora"}, 1),
            final("revise_task", revision_instructions="Add emojis"),
        ]
    )
    checkpointer, close = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, _, _, graph = service_for(
        model, Retriever(), checkpointer=checkpointer
    )

    try:
        service.process_message("chat-a", "Write Skyrim")
        skyrim = active_task(graph)
        with pytest.raises(
            ConversationServiceError,
            match="Historical retrieval must not revise the active task",
        ):
            service.process_message("chat-a", "Remember Aurora and add emojis")

        assert len(writer.tasks) == 1
        assert active_task(graph) == skyrim
    finally:
        close()


def test_search_only_can_clarify_but_cannot_start_writing() -> None:
    context = {
        "reason": "ambiguous_candidates",
        "candidate_summaries": ["2026-03-01 Writer — Aurora", "2026-03-02 Editor — Aurora"],
        "recommended_question": "Which Aurora draft should I use?",
    }
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "Aurora"}, 1),
            final("chat", talker_context=context),
        ]
    )
    retriever = Retriever(
        results=(
            result(1, "artifact-one", "Aurora first candidate"),
            result(2, "artifact-two", "Aurora second candidate"),
        )
    )
    service, writer, talker, store, _ = service_for(model, retriever)

    service.process_message("chat-a", "Use our Aurora draft")

    assert not writer.tasks and not store.runs
    assert talker.contexts[0].recommended_question == "Which Aurora draft should I use?"
    output = model.invocations[1]
    assert isinstance(output[-1], ToolMessage)
    assert '"artifact_id": "artifact-one"' in output[-1].content
    assert '"artifact_id": "artifact-two"' in output[-1].content


def test_date_arguments_and_runtime_timezone_are_model_controlled() -> None:
    model = ScriptedChatModel(
        [
            call(
                "search_corpus",
                {
                    "query": "weekly post",
                    "created_from": "2026-03-28T23:00:00Z",
                    "created_to": "2026-03-29T21:59:59Z",
                    "prefer_recent": True,
                },
                1,
            ),
            final("chat"),
        ]
    )
    retriever = Retriever()
    service, _, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "Find today's latest weekly post")

    assert retriever.searches[0]["created_from"].isoformat() == "2026-03-28T23:00:00+00:00"
    assert retriever.searches[0]["created_to"].isoformat() == "2026-03-29T21:59:59+00:00"
    assert retriever.searches[0]["prefer_recent"] is True
    prompt = model.invocations[0][0].content
    assert "Europe/Madrid" in prompt
    assert "2026-03-29T01:30:00+01:00" in prompt
    assert "2026-03-29T00:30:00+00:00" in prompt


def test_runtime_date_is_dynamic_and_uses_summer_daylight_saving_offset() -> None:
    summer = datetime(2026, 7, 5, 10, 15, tzinfo=UTC)
    model = ScriptedChatModel([final("chat")])
    service, _, _, _, _ = service_for(
        model,
        Retriever(),
        clock=lambda: summer,
    )

    service.process_message("chat-a", "hello in summer")

    prompt = model.invocations[0][0].content
    assert "2026-07-05T12:15:00+02:00" in prompt
    assert "2026-07-05T10:15:00+00:00" in prompt
    assert "2026-03-29" not in prompt


def test_invalid_arguments_are_structured_and_model_can_refine() -> None:
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "Aurora", "conversation_id": "leak"}, 1),
            call("search_corpus", {"query": "Aurora refined"}, 2),
            final("chat"),
        ]
    )
    retriever = Retriever()
    service, _, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "Find Aurora")

    first_error = model.invocations[1][-1]
    assert isinstance(first_error, ToolMessage)
    assert "invalid_tool_arguments" in first_error.content
    assert [item["query"] for item in retriever.searches] == ["Aurora refined"]
    properties = model.bindings[0][0].args_schema.model_json_schema()["properties"]
    assert "conversation_id" not in properties


def test_cross_conversation_get_is_missing_and_loop_is_bounded() -> None:
    draft = RetrievedDraft(
        "artifact-b",
        "task-b",
        ArtifactProducer.WRITER,
        NOW,
        "chat-b",
        "request",
        "secret B content",
    )
    missing_model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "draft B"}, 1),
            call("get_draft", {"artifact_id": "artifact-b"}, 2),
            final("chat"),
        ]
    )
    service, writer, _, _, _ = service_for(missing_model, Retriever(draft))
    service.process_message("chat-a", "get it")
    assert not writer.tasks
    assert "artifact_not_found" in missing_model.invocations[2][-1].content

    loop_model = ScriptedChatModel(
        [call("search_corpus", {"query": f"q-{index}"}, index) for index in range(7)]
    )
    loop_service, _, _, _, _ = service_for(loop_model, Retriever())
    with pytest.raises(ConversationServiceError, match="Coordinator tool limit exceeded"):
        loop_service.process_message("chat-a", "keep searching")


def test_no_tool_chat_and_supplied_writing_take_existing_routes() -> None:
    model = ScriptedChatModel(
        [
            final("chat"),
            final("start_writing_task", task_input="Write directly supplied topic"),
        ]
    )
    retriever = Retriever()
    service, writer, talker, store, _ = service_for(model, retriever)

    service.process_message("chat-a", "hello")
    service.process_message("chat-a", "Write directly supplied topic")

    assert len(talker.contexts) == 1
    assert writer.tasks[0].working_draft is None
    assert len(store.runs) == 1
    assert retriever.searches == [] and retriever.gets == []
    assert len(model.invocations) == 2


def test_gemini_block_no_tool_chat_and_supplied_writing_complete_end_to_end() -> None:
    model = ScriptedChatModel(
        [
            block_final("chat"),
            block_final("start_writing_task", task_input="Write supplied topic"),
        ]
    )
    retriever = Retriever()
    service, writer, talker, store, _ = service_for(model, retriever)

    service.process_message("chat-a", "hello")
    service.process_message("chat-a", "Write supplied topic")

    assert len(talker.contexts) == 1
    assert writer.tasks[0].brief.original_request == "Write supplied topic"
    assert len(store.runs) == 1
    assert retriever.searches == [] and retriever.gets == []


def test_gemini_block_chat_and_revision_with_active_reviewed_task_complete(
    tmp_path: Path,
) -> None:
    model = ScriptedChatModel(
        [
            block_final("start_writing_task", task_input="Write supplied topic"),
            block_final("chat"),
            block_final("revise_task", revision_instructions="Make it shorter"),
        ]
    )
    checkpointer, close_checkpointer = create_sqlite_checkpointer(tmp_path / "state.db")
    service, writer, talker, store, graph = service_for(
        model, Retriever(), checkpointer=checkpointer
    )

    service.process_message("chat-a", "Write supplied topic")
    task_after_writing = graph.get_state(
        {"configurable": {"thread_id": "editorial:v1:chat-a"}}
    ).values["conversation"].active_task
    assert task_after_writing is not None
    service.process_message("chat-a", "hello")
    task_after_chat = graph.get_state(
        {"configurable": {"thread_id": "editorial:v1:chat-a"}}
    ).values["conversation"].active_task
    assert task_after_chat == task_after_writing
    service.process_message("chat-a", "Make it shorter")

    assert len(talker.contexts) == 1
    assert len(writer.tasks) == 2
    assert writer.tasks[1].brief.instructions == ("Make it shorter",)
    assert len(store.runs) == 2
    close_checkpointer()


def test_tool_call_blocks_still_enter_tool_path_before_final_parsing() -> None:
    tool_response = call("search_corpus", {"query": "Aurora"}, 1)
    tool_response.content = [
        {"type": "thinking", "thinking": "tool selection", "extras": {"signature": "sig"}}
    ]
    model = ScriptedChatModel([tool_response, block_final("chat")])
    retriever = Retriever()
    service, _, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "Find Aurora")

    assert [search["query"] for search in retriever.searches] == ["Aurora"]
    assert model.invocations[1][-2] is tool_response
    assert tool_response.content[0]["extras"]["signature"] == "sig"  # type: ignore[index]


def test_block_final_without_usable_text_fails_through_sanitized_boundary() -> None:
    model = ScriptedChatModel([AIMessage(content=[{"type": "thinking", "thinking": "only"}])])
    service, _, _, _, _ = service_for(model, Retriever())

    with pytest.raises(ConversationServiceError, match="Coordinator failed"):
        service.process_message("chat-a", "hello")


def test_get_draft_before_search_returns_error_without_calling_retriever() -> None:
    model = ScriptedChatModel(
        [call("get_draft", {"artifact_id": "artifact-old"}, 1), final("chat")]
    )
    retriever = Retriever()
    service, writer, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "Use an old draft")

    assert retriever.gets == []
    assert "search_required" in model.invocations[1][-1].content
    assert writer.tasks == []


def test_search_excerpt_alone_cannot_initialize_writing_task() -> None:
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "Aurora"}, 1),
            final("start_writing_task", task_input="Rewrite it"),
        ]
    )
    service, writer, _, store, _ = service_for(model, Retriever())

    with pytest.raises(ConversationServiceError, match="complete historical draft"):
        service.process_message("chat-a", "Rewrite our Aurora draft")

    assert writer.tasks == [] and store.runs == []


def test_multiple_refined_searches_are_model_selected_and_executed() -> None:
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "launch"}, 1),
            call("search_corpus", {"query": "Aurora launch", "top_k": 2}, 2),
            final("chat"),
        ]
    )
    retriever = Retriever()
    service, _, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "Find the launch draft")

    assert [search["query"] for search in retriever.searches] == [
        "launch",
        "Aurora launch",
    ]
    assert retriever.searches[1]["top_k"] == 2


def test_no_match_context_reaches_talker_without_editorial_run() -> None:
    context = {
        "reason": "no_match",
        "candidate_summaries": [],
        "recommended_question": "Could you provide another clue or paste the text?",
    }
    model = ScriptedChatModel(
        [call("search_corpus", {"query": "missing"}, 1), final("chat", talker_context=context)]
    )
    service, writer, talker, store, _ = service_for(model, Retriever())

    service.process_message("chat-a", "Use the missing draft")

    assert talker.contexts[0].reason.value == "no_match"
    assert talker.contexts[0].candidate_summaries == ()
    assert writer.tasks == [] and store.runs == []


def test_completed_tool_turn_clears_all_retrieval_state() -> None:
    model = ScriptedChatModel(
        [call("search_corpus", {"query": "missing"}, 1), final("chat")]
    )
    service, _, _, _, graph = service_for(
        model,
        Retriever(),
        checkpointer=InMemorySaver(),
    )

    service.process_message("chat-a", "Find a missing draft")
    snapshot = graph.get_state({"configurable": {"thread_id": "editorial:v1:chat-a"}})

    for key in (
        "coordinator_messages",
        "coordinator_tool_steps",
        "coordinator_search_completed",
        "retrieved_draft",
    ):
        assert snapshot.values.get(key) is None


def test_interleaved_conversation_turns_keep_tool_scope_isolated() -> None:
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "first A"}, 1),
            final("chat"),
            call("search_corpus", {"query": "only B"}, 2),
            final("chat"),
            call("search_corpus", {"query": "second A"}, 3),
            final("chat"),
        ]
    )
    retriever = Retriever()
    service, _, _, _, _ = service_for(model, retriever)

    service.process_message("chat-a", "first")
    service.process_message("chat-b", "second")
    service.process_message("chat-a", "third")

    assert [search["conversation_id"] for search in retriever.searches] == [
        "chat-a",
        "chat-b",
        "chat-a",
    ]


def test_langchain_tool_messages_survive_sqlite_checkpoint_restart(tmp_path: Path) -> None:
    database = tmp_path / "tool-loop.db"
    saver, close = create_sqlite_checkpointer(database)
    first_model = ScriptedChatModel(
        [call("search_corpus", {"query": "missing"}, 1), final("chat")]
    )
    first, _, _, _, _ = service_for(first_model, Retriever(), checkpointer=saver)
    first.process_message("chat-a", "Find it")
    assert isinstance(first_model.invocations[1][-1], ToolMessage)
    close()

    restored_saver, restored_close = create_sqlite_checkpointer(database)
    second_model = ScriptedChatModel([final("chat")])
    second, _, talker, _, graph = service_for(
        second_model,
        Retriever(),
        checkpointer=restored_saver,
    )
    second.process_message("chat-a", "hello after restart")
    snapshot = graph.get_state({"configurable": {"thread_id": "editorial:v1:chat-a"}})

    assert len(talker.contexts) == 1
    assert len(snapshot.values["conversation"].recent_messages) == 4
    restored_close()


def test_retrieved_revise_path_persists_writer_and_editor_outputs() -> None:
    original = RetrievedDraft(
        "artifact-old",
        "task-old",
        ArtifactProducer.EDITOR,
        NOW,
        "chat-a",
        "old request",
        "old complete draft",
    )
    model = ScriptedChatModel(
        [
            call("search_corpus", {"query": "old"}, 1),
            call("get_draft", {"artifact_id": "artifact-old"}, 2),
            final("start_writing_task", task_input="Improve the old draft"),
        ]
    )
    service, _, _, store, _ = service_for(
        model,
        Retriever(original),
        critic_verdict=CriticVerdict.REVISE,
    )

    service.process_message("chat-a", "Improve our old draft")

    assert {artifact.producer for artifact in store.runs[0]} == {
        ArtifactProducer.WRITER,
        ArtifactProducer.EDITOR,
    }
    assert {artifact.content for artifact in store.runs[0]} == {
        "rewritten:old complete draft",
        "edited historical output",
    }
    assert original.content == "old complete draft"
