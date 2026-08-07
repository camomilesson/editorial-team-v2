from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.entities import SpanType

import editorial_team.evaluation.agent_harness as harness
import editorial_team.evaluation.agent_real_runner as real_runner
from editorial_team.artifacts import (
    ArtifactProducer,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
)
from editorial_team.contracts.identity import validate_identifier
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.evaluation.agent_cases import (
    AgentEvaluationCase,
    CaseSetup,
    OutcomeExpectation,
    ParameterExpectation,
    load_agent_evaluation_cases,
)
from editorial_team.evaluation.agent_harness import (
    EVALUATION_AGENT_TEMPERATURE,
    AgentInvocation,
    AgentRunFailure,
    compare_parameters,
    evaluate_goal,
    run_agent_evaluation,
    run_identities,
)
from editorial_team.evaluation.generation_judges import JudgeScore
from editorial_team.evaluation.generation_models import GenerationContext
from editorial_team.evaluation.trace_adapters import (
    EvaluationToolCall,
    GenerationReference,
    RetrievalReference,
)

EXPECTED_CASES = {
    "chat_simple",
    "write_from_prompt",
    "write_with_memory",
    "retrieve_exact_draft",
    "retrieve_latest_topic",
    "retrieve_recent_period",
    "active_revision",
    "historical_revision",
    "search_no_match",
    "retrieval_rerank",
    "ambiguous_reference",
    "tool_restraint",
}


def _trace(
    tool_names: tuple[str, ...], *, search_arguments: dict[str, object] | None = None
) -> object:
    root = SimpleNamespace(
        parent_id=None,
        span_type=SpanType.AGENT,
        start_time_ns=1,
        span_id="root",
        name="root",
        attributes={"eval_case_id": "unit"},
    )
    spans = [root]
    for index, name in enumerate(tool_names, 1):
        arguments = (
            search_arguments
            or {
                "query": "aurora",
                "created_from": None,
                "created_to": None,
                "prefer_recent": False,
                "top_k": 5,
                "rerank": False,
            }
            if name == "search_corpus"
            else {"artifact_id": "artifact-1"}
        )
        spans.append(
            SimpleNamespace(
                parent_id="root",
                span_type=SpanType.TOOL,
                start_time_ns=index + 1,
                span_id=f"tool-{index}",
                name=name,
                attributes={"tool.name": name, "tool.arguments": arguments},
            )
        )
    return SimpleNamespace(data=SimpleNamespace(spans=spans))


def test_fixed_case_plan_contains_all_twelve_declarative_cases() -> None:
    cases = load_agent_evaluation_cases()

    assert len(cases) == 12
    assert {case.case_id for case in cases} == EXPECTED_CASES
    assert all(case.input_message.strip() for case in cases)
    assert all(case.accepted_trajectories for case in cases)
    assert all(case.outcome.description.strip() for case in cases)
    ambiguous = next(case for case in cases if case.case_id == "ambiguous_reference")
    assert ambiguous.acceptable_alternatives == (("search_corpus", "get_draft"),)
    historical = next(case for case in cases if case.case_id == "historical_revision")
    assert historical.setup.active_task is not None
    assert historical.outcome.required_facts == (
        "historical_orbit_selected",
        "ember_not_used_as_source",
    )
    active = next(case for case in cases if case.case_id == "active_revision")
    assert active.setup.active_task is not None
    assert active.setup.active_task.working_draft
    memory = next(case for case in cases if case.case_id == "write_with_memory")
    assert memory.expected_trajectory == ("search_corpus", "get_draft")
    recent = next(case for case in cases if case.case_id == "retrieve_recent_period")
    assert recent.expected_trajectory == ("search_corpus", "get_draft")
    no_match = next(case for case in cases if case.case_id == "search_no_match")
    assert no_match.acceptable_alternatives == (("search_corpus", "search_corpus"),)
    rerank = next(case for case in cases if case.case_id == "retrieval_rerank")
    assert rerank.expected_trajectory == ("search_corpus", "get_draft")
    assert all(item.field != "rerank" for item in rerank.parameter_expectations)
    assert all(not case.score_generation or case.generation_golden_answer for case in cases)


def test_exactly_three_valid_unique_isolated_run_slots_per_case() -> None:
    identities = run_identities(load_agent_evaluation_cases())

    assert len(identities) == 36
    assert len({item.conversation_id for item in identities}) == 36
    assert len({item.thread_id for item in identities}) == 36
    assert all(item.request_origin == "batch" for item in identities)
    assert all(
        validate_identifier(item.conversation_id, "conversation_id") == item.conversation_id
        for item in identities
    )
    assert all(":" not in item.conversation_id for item in identities)
    assert all("/" not in item.conversation_id for item in identities)
    assert all("\\" not in item.conversation_id for item in identities)
    assert all(item.thread_id == f"editorial:v1:{item.conversation_id}" for item in identities)
    assert identities[0].conversation_id == "eval-chat-simple-r1"
    for case_id in EXPECTED_CASES:
        assert [item.run_number for item in identities if item.case_id == case_id] == [1, 2, 3]
    with pytest.raises(ValueError, match="exactly three"):
        run_identities(load_agent_evaluation_cases(), runs_per_case=2)


def test_fixture_setup_and_measured_invocation_share_run_identity(tmp_path: Path) -> None:
    cases = load_agent_evaluation_cases()
    case = next(item for item in cases if item.case_id == "write_with_memory")
    identity = next(item for item in run_identities(cases) if item.case_id == case.case_id)
    artifact_path = tmp_path / "artifacts.db"

    real_runner._seed_artifacts(artifact_path, case, identity.conversation_id)
    store = SQLiteArtifactStore(artifact_path, chunker=ParagraphChunker())
    store.initialize()
    try:
        artifact = store.get_artifact(
            artifact_id_for("eval-aurora-memory", ArtifactProducer.WRITER)
        )
    finally:
        store.close()
    assert artifact.conversation_id == identity.conversation_id

    processed: list[tuple[str, str, int]] = []

    class Service:
        _graph_runner = SimpleNamespace(
            get_state=lambda _config: SimpleNamespace(values={"conversation": None})
        )

        def process_message(
            self, conversation_id: str, _message: str, **metadata: object
        ) -> tuple[object, ...]:
            processed.append(
                (
                    conversation_id,
                    str(metadata["eval_case_id"]),
                    int(metadata["eval_run_number"]),
                )
            )
            return ()

    real_runner._invoke_case(Service(), case, identity, agent_temperature=0.2)
    assert processed == [(identity.conversation_id, case.case_id, identity.run_number)]


def test_active_fixture_seeds_a_substantive_reviewed_domain_task() -> None:
    case = next(item for item in load_agent_evaluation_cases() if item.case_id == "active_revision")
    identity = next(item for item in run_identities((case,)) if item.run_number == 1)
    updates: list[tuple[object, object]] = []
    service = SimpleNamespace(
        _graph_runner=SimpleNamespace(
            update_state=lambda config, values: updates.append((config, values))
        )
    )

    real_runner._seed_active_state(service, identity, case.setup.active_task)

    assert len(updates) == 1
    config, values = updates[0]
    assert config == {"configurable": {"thread_id": identity.thread_id}}
    task = values["conversation"].active_task
    assert task.conversation_id == identity.conversation_id
    assert task.status is WritingTaskStatus.REVIEWED
    assert task.working_draft == case.setup.active_task.working_draft
    assert len(task.working_draft.split()) >= 15


def test_exact_alternative_and_empty_trajectory_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        AgentEvaluationCase("empty", "hello", CaseSetup(), (), (), OutcomeExpectation("ok")),
        AgentEvaluationCase(
            "alternative",
            "hello",
            CaseSetup(),
            ("search_corpus",),
            (("search_corpus", "get_draft"),),
            OutcomeExpectation("ok"),
        ),
    )
    used_traces: list[object] = []

    class Executor:
        def execute(
            self, case: AgentEvaluationCase, identity: object, **kwargs: object
        ) -> AgentInvocation:
            del identity, kwargs
            tools = () if case.case_id == "empty" else ("search_corpus", "get_draft")
            trace = _trace(tools)
            used_traces.append(trace)
            return AgentInvocation(trace, f"trace-{len(used_traces)}", "response", {})

    original = harness.trace_to_tool_calls
    adapter_calls: list[object] = []

    def tracked(trace: object) -> list[EvaluationToolCall]:
        adapter_calls.append(trace)
        return original(trace)

    monkeypatch.setattr(harness, "trace_to_tool_calls", tracked)
    results = run_agent_evaluation(cases, Executor())

    assert len(results) == 6
    assert all(result.trajectory_passed for result in results)
    assert adapter_calls == used_traces
    assert {result.request_origin for result in results} == {"batch"}
    assert {result.agent_temperature for result in results} == {EVALUATION_AGENT_TEMPERATURE}


def test_failed_invocation_preserves_persisted_trace_id() -> None:
    case = AgentEvaluationCase("failure", "input", CaseSetup(), (), (), OutcomeExpectation("fails"))

    class Executor:
        def execute(self, *args: object, **kwargs: object) -> AgentInvocation:
            del args, kwargs
            raise AgentRunFailure("tr-persisted", "ConversationServiceError: Coordinator failed")

    results = run_agent_evaluation((case,), Executor())

    assert len(results) == 3
    assert {result.trace_id for result in results} == {"tr-persisted"}
    assert {result.error for result in results} == {"ConversationServiceError: Coordinator failed"}


def test_failed_trace_lookup_joins_case_and_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def stored(trace_id: str, case_id: str, run_number: int) -> object:
        return SimpleNamespace(
            info=SimpleNamespace(trace_id=trace_id, tags={"eval_case_id": case_id}),
            data=SimpleNamespace(
                spans=(
                    SimpleNamespace(
                        parent_id=None,
                        attributes={"evaluation.run_number": run_number},
                    ),
                )
            ),
        )

    traces = (
        stored("tr-old", "active_revision", 2),
        stored("tr-wrong-run", "active_revision", 1),
        stored("tr-correct", "active_revision", 2),
    )
    monkeypatch.setattr(real_runner.mlflow, "search_traces", lambda **_kwargs: traces)
    monkeypatch.setattr(
        real_runner,
        "Trace",
        SimpleNamespace(from_json=lambda _payload: traces[2]),
    )
    traces[2].to_json = lambda: "{}"

    trace = real_runner._new_stored_trace("experiment", {"tr-old"}, "active_revision", 2)

    assert trace.info.trace_id == "tr-correct"


def test_active_and_historical_facts_require_substantive_state_changes() -> None:
    timestamp = datetime(2026, 8, 7, 12, tzinfo=UTC)
    report = CriticReport(CriticVerdict.PASS, "fixture")

    def task(request: str, draft: str, *instructions: str) -> WritingTask:
        return WritingTask(
            "eval-task",
            "eval-active-revision-r1",
            WritingBrief(request, instructions),
            WritingTaskStatus.REVIEWED,
            timestamp,
            timestamp,
            draft,
            report,
        )

    ember = task("Write the Ember update", "Long factual Ember opening.")
    revised = task(
        "Write the Ember update",
        "Factual Ember opening.",
        "Make the opening shorter while preserving its facts.",
    )
    orbit = task("Write the Orbit launch draft", "A revised Orbit introduction.")

    assert real_runner._active_revision_applied(ember, revised)
    assert not real_runner._active_revision_applied(ember, ember)
    assert real_runner._historical_orbit_selected(orbit)
    assert real_runner._ember_not_used_as_source(ember, orbit)


def test_corrected_write_with_memory_preserves_real_two_of_three_pattern() -> None:
    case = next(
        case for case in load_agent_evaluation_cases() if case.case_id == "write_with_memory"
    )
    recorded_prefer_recent = iter((False, False, True))

    class Executor:
        def execute(self, *args: object, **kwargs: object) -> AgentInvocation:
            del args, kwargs
            prefer_recent = next(recorded_prefer_recent)
            trace = _trace(
                ("search_corpus", "get_draft"),
                search_arguments={
                    "query": "Aurora notes",
                    "created_from": None,
                    "created_to": None,
                    "prefer_recent": prefer_recent,
                    "top_k": 5,
                    "rerank": False,
                },
            )
            return AgentInvocation(
                trace,
                "trace",
                "September 14 Quiet Momentum",
                {},
                RetrievalReference("write_with_memory", frozenset({"chunk"})),
                GenerationReference("write_with_memory", "golden", ()),
            )

    monkey_case = replace(case, score_retrieval=False, score_generation=False)
    results = run_agent_evaluation((monkey_case,), Executor())

    assert [result.trajectory_passed for result in results] == [True, True, True]
    assert [result.parameters_passed for result in results] == [True, True, False]


def test_parameter_comparison_has_narrow_field_semantics() -> None:
    calls = [
        EvaluationToolCall(
            "search_corpus",
            {
                "query": "  Aurora   Product Launch ",
                "created_from": "2026-07-27T00:00:00+00:00",
                "created_to": None,
                "prefer_recent": True,
                "top_k": 5,
                "rerank": True,
            },
        )
    ]
    comparisons = compare_parameters(
        calls,
        (
            ParameterExpectation(0, "query", "contains_terms", ("aurora", "launch")),
            ParameterExpectation(0, "top_k", "equals", 5),
            ParameterExpectation(0, "rerank", "equals", False),
        ),
    )

    assert [item.passed for item in comparisons] == [True, True, False]


def test_goal_completion_is_independent_of_trajectory() -> None:
    case = AgentEvaluationCase(
        "unit",
        "input",
        CaseSetup(),
        (),
        (),
        OutcomeExpectation("contains answer", required_response_terms=("answer",)),
    )
    invocation = AgentInvocation(_trace(("get_draft",)), "trace", "The answer is here", {})

    assert evaluate_goal(case, invocation) is True
    assert tuple(call.tool for call in harness.trace_to_tool_calls(invocation.trace)) != (
        case.expected_trajectory
    )
    assert (
        evaluate_goal(
            replace(
                case, outcome=OutcomeExpectation("missing", required_response_terms=("absent",))
            ),
            invocation,
        )
        is False
    )


def test_real_runner_paths_are_evaluation_scoped() -> None:
    source = Path("src/editorial_team/evaluation/agent_real_runner.py").read_text(encoding="utf-8")
    assert "evaluation/agent/.runtime" in source
    assert "runtime_data" not in source


def test_hw2_adapters_and_existing_scorers_are_exercised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = AgentEvaluationCase(
        "scored",
        "input",
        CaseSetup(),
        (),
        (),
        OutcomeExpectation("ok"),
        score_retrieval=True,
        score_generation=True,
        golden_fixture_ids=("fixture",),
    )
    context = GenerationContext("chunk-1", "artifact-1", "context")
    retrieval_reference = RetrievalReference("scored", frozenset({"chunk-1"}))
    generation_reference = GenerationReference("scored", "golden", (context,))
    retrieval_calls: list[object] = []
    generation_calls: list[object] = []

    def retrieval_adapter(trace: object, references: object) -> object:
        retrieval_calls.append((trace, references))
        return SimpleNamespace(predictions=("chunk-1",), golden=frozenset({"chunk-1"}))

    def generation_adapter(trace: object, references: object) -> object:
        generation_calls.append((trace, references))
        return SimpleNamespace(
            query="query",
            candidate_answer="answer",
            golden_answer="golden",
            retrieved_contexts=(context,),
            golden_contexts=(context,),
        )

    monkeypatch.setattr(harness, "trace_to_retrieval_scorer_input", retrieval_adapter)
    monkeypatch.setattr(harness, "trace_to_generation_judge_input", generation_adapter)

    class Executor:
        def execute(self, *args: object, **kwargs: object) -> AgentInvocation:
            del args, kwargs
            return AgentInvocation(
                _trace(()),
                "trace",
                "answer",
                {},
                retrieval_reference,
                generation_reference,
            )

    class Judge:
        def judge(self, *args: object, **kwargs: object) -> JudgeScore:
            del args, kwargs
            return JudgeScore(0.75, "bounded test")

    results = run_agent_evaluation(
        (case,),
        Executor(),
        generation_judge=Judge(),  # type: ignore[arg-type]
    )

    assert len(retrieval_calls) == 3
    assert len(generation_calls) == 3
    assert all(
        result.retrieval_scores is not None and result.retrieval_scores["recall_at_5"] == 1.0
        for result in results
    )
    assert all(
        result.generation_scores is not None and result.generation_scores["faithfulness"] == 0.75
        for result in results
    )
