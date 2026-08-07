from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.entities import SpanType

import editorial_team.evaluation.agent_harness as harness
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


def _trace(tool_names: tuple[str, ...]) -> object:
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
            {
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
    assert historical.setup.protected_state_labels == ("unrelated_active_task",)
    assert "unrelated_active_task_preserved" in historical.outcome.required_facts


def test_exactly_three_unique_isolated_run_slots_per_case() -> None:
    identities = run_identities(load_agent_evaluation_cases())

    assert len(identities) == 36
    assert len({item.conversation_id for item in identities}) == 36
    assert len({item.thread_id for item in identities}) == 36
    assert all(item.request_origin == "batch" for item in identities)
    assert all(item.case_id in item.conversation_id for item in identities)
    for case_id in EXPECTED_CASES:
        assert [item.run_number for item in identities if item.case_id == case_id] == [1, 2, 3]
    with pytest.raises(ValueError, match="exactly three"):
        run_identities(load_agent_evaluation_cases(), runs_per_case=2)


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
