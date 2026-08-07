from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.entities import SpanType

import editorial_team.evaluation.agent_reporting as reporting
from editorial_team.evaluation.agent_cases import load_agent_evaluation_cases
from editorial_team.evaluation.agent_harness import (
    AgentRunResult,
    ParameterComparison,
    write_results,
)
from editorial_team.evaluation.agent_reporting import (
    GENERATION_FEEDBACK_NAMES,
    PART1_FEEDBACK_NAMES,
    RETRIEVAL_FEEDBACK_NAMES,
    CampaignManifest,
    TraceLocation,
    aggregate_campaign,
    load_run_results,
    log_campaign_feedback,
    log_campaign_safety_feedback,
    rescore_part1_from_stored_traces,
    rescore_stored_traces,
)
from editorial_team.mlflow_tracing import (
    ATTR_CANDIDATE_ANSWER,
    ATTR_EVAL_CASE_ID,
    ATTR_REQUEST_ORIGIN,
    ATTR_RETRIEVAL_CONTEXTS,
    ATTR_RETRIEVAL_FINAL_RESULTS,
    ATTR_RETRIEVAL_REQUEST,
)
from editorial_team.safety import (
    ATTR_INPUT_BLOCKED,
    ATTR_PREFLIGHT_FLAGGED,
    ATTR_SAFETY_SCHEMA,
    SAFETY_SCHEMA_VERSION,
)


def _result(
    case_id: str,
    run_number: int,
    *,
    trajectory: bool = True,
    parameters: bool = True,
    goal: bool = True,
    comparisons: tuple[ParameterComparison, ...] = (),
    retrieval: dict[str, float] | None = None,
    generation: dict[str, float] | None = None,
    error: str | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        case_id=case_id,
        run_number=run_number,
        conversation_id=f"eval-{case_id}-r{run_number}",
        thread_id=f"editorial:v1:eval-{case_id}-r{run_number}",
        request_origin="batch",
        agent_temperature=0.2,
        trace_id=f"tr-{case_id}-{run_number}",
        tool_trajectory=("search_corpus",) if trajectory else (),
        accepted_trajectories=(("search_corpus",),),
        trajectory_passed=trajectory,
        parameter_comparisons=comparisons,
        parameters_passed=parameters,
        goal_completion_passed=goal,
        retrieval_scores=retrieval,
        generation_scores=generation,
        error=error,
    )


def _three(case_id: str, passed: tuple[bool, bool, bool]) -> tuple[AgentRunResult, ...]:
    return tuple(_result(case_id, index, parameters=value) for index, value in enumerate(passed, 1))


@pytest.mark.parametrize(
    ("passed", "pattern", "rate", "pass_at_3", "pass_power_3"),
    [
        ((True, True, True), "3/3", 1.0, 1, 1),
        ((True, True, False), "2/3", 2 / 3, 1, 0),
        ((False, False, False), "0/3", 0.0, 0, 0),
    ],
)
def test_three_run_reliability_formulas(
    passed: tuple[bool, bool, bool],
    pattern: str,
    rate: float,
    pass_at_3: int,
    pass_power_3: int,
) -> None:
    case = aggregate_campaign(_three("case", passed)).cases[0]

    assert case.pattern == pattern
    assert case.success_rate == rate
    assert case.pass_at_3 == pass_at_3
    assert case.pass_power_3 == pass_power_3


def test_final_reliability_requires_exactly_runs_one_two_and_three() -> None:
    with pytest.raises(ValueError, match="exactly runs 1, 2, and 3"):
        aggregate_campaign(_three("case", (True, True, True))[:2])
    duplicate = (_result("case", 1), _result("case", 2), _result("case", 2))
    with pytest.raises(ValueError, match="exactly runs 1, 2, and 3"):
        aggregate_campaign(duplicate)


def test_write_with_memory_two_of_three_emerges_from_recorded_parameters() -> None:
    observed = (False, False, True)
    results = tuple(
        _result(
            "write_with_memory",
            run_number,
            parameters=prefer_recent is False,
            comparisons=(
                ParameterComparison(
                    0, "prefer_recent", prefer_recent is False, False, prefer_recent
                ),
            ),
        )
        for run_number, prefer_recent in enumerate(observed, 1)
    )

    summary = aggregate_campaign(results)

    assert summary.cases[0].pattern == "2/3"
    assert [run.overall_passed for run in summary.runs] == [True, True, False]


def test_tool_parameter_and_goal_metrics_are_independent() -> None:
    results = (
        _result("case", 1, trajectory=True, parameters=False, goal=True),
        _result("case", 2, trajectory=False, parameters=True, goal=True),
        _result("case", 3, trajectory=True, parameters=True, goal=False),
    )

    suite = aggregate_campaign(results).suite

    assert suite.tool_selection_accuracy == 2 / 3
    assert suite.run_level_parameter_accuracy == 2 / 3
    assert suite.goal_completion_rate == 2 / 3
    assert suite.overall_successful_runs == 0


def test_no_tool_runs_do_not_enter_field_denominator() -> None:
    comparisons = (
        ParameterComparison(0, "query", True, "x", "x"),
        ParameterComparison(0, "top_k", False, 5, 3),
    )
    results = (
        _result("with-tools", 1, parameters=False, comparisons=comparisons),
        _result("with-tools", 2, parameters=False, comparisons=comparisons),
        _result("with-tools", 3, parameters=False, comparisons=comparisons),
        _result("no-tools", 1),
        _result("no-tools", 2),
        _result("no-tools", 3),
    )

    suite = aggregate_campaign(results).suite

    assert suite.field_level_parameter_accuracy == 0.5
    assert suite.run_level_parameter_accuracy == 0.5


def test_feedback_logs_part1_and_unchanged_hw2_values_to_origin_trace() -> None:
    result = _result(
        "case",
        1,
        retrieval={"mrr_at_5": 0.75, "precision_at_5": 0.4, "recall_at_5": 1.0},
        generation={
            "faithfulness": 0.8,
            "answer_relevance": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
        },
    )
    logged: list[dict[str, object]] = []
    manifest = CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", "results.json")

    count = log_campaign_feedback(
        (result,),
        manifest,
        get_trace=lambda trace_id, **_kwargs: SimpleNamespace(
            info=SimpleNamespace(trace_id=trace_id, experiment_id="exp-1")
        ),
        log_feedback=lambda **kwargs: logged.append(kwargs),
    )

    assert count == 11
    assert {item["trace_id"] for item in logged} == {result.trace_id}
    values = {str(item["name"]): item["value"] for item in logged}
    assert values == {
        PART1_FEEDBACK_NAMES["tool_selection"]: 1.0,
        PART1_FEEDBACK_NAMES["tool_parameters"]: 1.0,
        PART1_FEEDBACK_NAMES["goal_completion"]: 1.0,
        PART1_FEEDBACK_NAMES["overall_pass"]: 1.0,
        RETRIEVAL_FEEDBACK_NAMES["mrr_at_5"]: 0.75,
        RETRIEVAL_FEEDBACK_NAMES["precision_at_5"]: 0.4,
        RETRIEVAL_FEEDBACK_NAMES["recall_at_5"]: 1.0,
        GENERATION_FEEDBACK_NAMES["faithfulness"]: 0.8,
        GENERATION_FEEDBACK_NAMES["answer_relevance"]: 0.9,
        GENERATION_FEEDBACK_NAMES["context_precision"]: 0.7,
        GENERATION_FEEDBACK_NAMES["context_recall"]: 0.6,
    }
    assert "secret-canary" not in repr(logged)


def test_feedback_works_after_result_serialization_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_results(path, (_result("case", 1),))
    reloaded = load_run_results(path)
    logged: list[dict[str, object]] = []

    log_campaign_feedback(
        reloaded,
        CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", str(path)),
        get_trace=lambda trace_id, **_kwargs: SimpleNamespace(
            info=SimpleNamespace(trace_id=trace_id, experiment_id="exp-1")
        ),
        log_feedback=lambda **kwargs: logged.append(kwargs),
    )

    assert len(logged) == 4
    assert {item["trace_id"] for item in logged} == {"tr-case-1"}


def test_feedback_overrides_existing_metric_instead_of_logging_duplicate() -> None:
    result = _result("case", 1)
    existing = SimpleNamespace(
        name=PART1_FEEDBACK_NAMES["tool_selection"],
        valid=True,
        assessment_id="assessment-old",
    )
    trace = SimpleNamespace(info=SimpleNamespace(experiment_id="exp-1", assessments=[existing]))
    logged: list[dict[str, object]] = []
    overridden: list[dict[str, object]] = []

    log_campaign_feedback(
        (result,),
        CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", "results.json"),
        get_trace=lambda *_args, **_kwargs: trace,
        log_feedback=lambda **kwargs: logged.append(kwargs),
        override_feedback=lambda **kwargs: overridden.append(kwargs),
    )

    assert len(overridden) == 1
    assert overridden[0]["assessment_id"] == "assessment-old"
    assert all(item["name"] != PART1_FEEDBACK_NAMES["tool_selection"] for item in logged)


def test_wrong_tracking_store_or_experiment_fails_clearly() -> None:
    manifest = CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", "results.json")
    with pytest.raises(RuntimeError, match="missing from campaign tracking store"):
        log_campaign_feedback((_result("case", 1),), manifest, get_trace=lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="different experiment"):
        log_campaign_feedback(
            (_result("case", 1),),
            manifest,
            get_trace=lambda *_a, **_k: SimpleNamespace(
                info=SimpleNamespace(experiment_id="wrong")
            ),
        )


def test_feedback_uses_the_tracking_location_declared_for_each_trace() -> None:
    first = _result("case", 1)
    second = _result("case", 2)
    locations = {
        first.trace_id: TraceLocation("sqlite:///:memory:", "exp-1", "one"),
        second.trace_id: TraceLocation("sqlite:///second.db", "exp-2", "two"),
    }
    observed: list[tuple[str, str]] = []

    def get_trace(trace_id: str, **_kwargs: object) -> object:
        location = locations[trace_id]
        observed.append((trace_id, reporting.mlflow.get_tracking_uri()))
        return SimpleNamespace(info=SimpleNamespace(experiment_id=location.experiment_id))

    log_campaign_feedback(
        (first, second),
        CampaignManifest(
            1,
            "sqlite:///:memory:",
            "exp-1",
            "campaign",
            "results.json",
            trace_locations=locations,
        ),
        get_trace=get_trace,
        log_feedback=lambda **_kwargs: None,
    )

    assert observed == [
        (first.trace_id, "sqlite:///:memory:"),
        (second.trace_id, "sqlite:///second.db"),
    ]


def test_safety_feedback_attaches_bounded_scores_to_the_origin_trace() -> None:
    result = _result("safety-case", 1)
    trace = SimpleNamespace(
        info=SimpleNamespace(experiment_id="exp-1"),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    attributes={
                        ATTR_SAFETY_SCHEMA: SAFETY_SCHEMA_VERSION,
                        ATTR_PREFLIGHT_FLAGGED: True,
                        ATTR_INPUT_BLOCKED: True,
                    }
                )
            ]
        ),
    )
    logged: list[dict[str, object]] = []

    report = log_campaign_safety_feedback(
        (result,),
        CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", "results.json"),
        get_trace=lambda *_args, **_kwargs: trace,
        log_feedback=lambda **kwargs: logged.append(kwargs),
    )

    assert report.evaluated == 1
    assert report.flagged == 1
    assert report.assessments_logged == 3
    assert {item["trace_id"] for item in logged} == {result.trace_id}
    assert {item["name"] for item in logged} == {
        "safety.threat_detected",
        "safety.defense_effective",
        "safety.unsafe_behavior",
    }
    assert "secret-canary" not in repr(logged)


def test_stored_trace_rescoring_uses_current_references_without_agent_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        item for item in load_agent_evaluation_cases() if item.case_id == "write_with_memory"
    )
    retrieval_reference, generation_reference = reporting._case_references(case)
    context = generation_reference.golden_contexts[0]
    root = SimpleNamespace(
        parent_id=None,
        span_type=SpanType.AGENT,
        attributes={
            ATTR_EVAL_CASE_ID: case.case_id,
            ATTR_REQUEST_ORIGIN: "batch",
            ATTR_CANDIDATE_ANSWER: generation_reference.golden_answer,
        },
    )
    retriever = SimpleNamespace(
        parent_id="root",
        span_type=SpanType.RETRIEVER,
        start_time_ns=1,
        span_id="retriever",
        attributes={
            ATTR_RETRIEVAL_REQUEST: {"query": "Aurora notes"},
            ATTR_RETRIEVAL_FINAL_RESULTS: [{"chunk_id": context.chunk_id}],
            ATTR_RETRIEVAL_CONTEXTS: [
                {
                    "chunk_id": context.chunk_id,
                    "artifact_id": context.artifact_id,
                    "content": context.content,
                }
            ],
        },
    )
    trace = SimpleNamespace(
        info=SimpleNamespace(experiment_id="exp-1"),
        data=SimpleNamespace(spans=[root, retriever]),
    )
    monkeypatch.setattr(reporting.mlflow, "get_trace", lambda *_args, **_kwargs: trace)

    class Judge:
        def judge(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(score=0.625)

    rescored = rescore_stored_traces(
        (_result(case.case_id, 1),),
        (case,),
        CampaignManifest(1, "sqlite:///:memory:", "exp-1", "campaign", "results.json"),
        generation_judge=Judge(),
    )

    assert retrieval_reference.golden_chunk_ids == frozenset({context.chunk_id})
    assert rescored[0].retrieval_scores == {
        "precision_at_5": 0.2,
        "recall_at_5": 1.0,
        "mrr_at_5": 1.0,
    }
    assert rescored[0].generation_scores == {
        "faithfulness": 0.625,
        "answer_relevance": 0.625,
        "context_precision": 0.625,
        "context_recall": 0.625,
    }


def test_cedar_goal_uses_completed_transformation_not_artifact_name() -> None:
    case = next(
        item for item in load_agent_evaluation_cases() if item.case_id == "retrieve_exact_draft"
    )
    comparisons = (
        ParameterComparison(0, "query", True, ("cedar", "manifesto"), "Cedar manifesto"),
        ParameterComparison(1, "artifact_id", True, "expected", "expected"),
    )

    def trace(writer: str, verdict: str) -> object:
        return SimpleNamespace(
            data=SimpleNamespace(
                spans=[
                    SimpleNamespace(
                        parent_id=None,
                        attributes={
                            "evaluation.candidate_answer": (
                                f"✍️ Writer\n\n{writer}\n\n🔍 Critic\n\n"
                                f"Verdict: {verdict}\n\nSummary: review"
                            )
                        },
                    )
                ]
            )
        )

    assert reporting._frozen_exact_draft_goal(
        trace("Build patiently, publish warmly, and revise with evidence.", "PASS"),
        case,
        comparisons,
    )
    assert not reporting._frozen_exact_draft_goal(
        trace(
            "Cedar manifesto full draft. Build patiently, publish clearly, revise with evidence.",
            "PASS",
        ),
        case,
        comparisons,
    )
    assert not reporting._frozen_exact_draft_goal(
        trace("Cedar is mentioned but no accepted workflow completed.", "REVISE"),
        case,
        comparisons,
    )


def test_frozen_authoritative_results_rescore_to_correct_part1_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("evaluation/agent/final-results.json")
    if not path.exists():
        pytest.skip("frozen authoritative campaign artifact is not available")
    results = load_run_results(path)
    cases = load_agent_evaluation_cases()
    traces: dict[str, object] = {}
    for result in results:
        comparison_values = {
            (item.call_index, item.field): item.observed for item in result.parameter_comparisons
        }
        spans = [
            SimpleNamespace(
                parent_id=None,
                span_type=SpanType.AGENT,
                attributes={
                    "evaluation.candidate_answer": (
                        "✍️ Writer\n\nA warmer complete rewrite preserving the source facts."
                        "\n\n🔍 Critic\n\nVerdict: PASS\n\nSummary: accepted"
                        if result.case_id == "retrieve_exact_draft"
                        else "completed"
                    )
                },
            )
        ]
        for index, tool in enumerate(result.tool_trajectory):
            arguments = (
                {
                    "query": comparison_values.get((index, "query"), "historical draft"),
                    "created_from": comparison_values.get((index, "created_from")),
                    "created_to": comparison_values.get((index, "created_to")),
                    "prefer_recent": comparison_values.get((index, "prefer_recent"), False),
                    "top_k": comparison_values.get((index, "top_k"), 5),
                    "rerank": comparison_values.get((index, "rerank"), False),
                }
                if tool == "search_corpus"
                else {"artifact_id": comparison_values.get((index, "artifact_id"), "artifact")}
            )
            spans.append(
                SimpleNamespace(
                    parent_id="root",
                    span_type=SpanType.TOOL,
                    start_time_ns=index,
                    span_id=f"tool-{index}",
                    name=tool,
                    attributes={"tool.name": tool, "tool.arguments": arguments},
                )
            )
        traces[result.trace_id] = SimpleNamespace(data=SimpleNamespace(spans=spans))
    monkeypatch.setattr(
        reporting.mlflow,
        "get_trace",
        lambda trace_id, **_kwargs: traces[trace_id],
    )
    rescored = rescore_part1_from_stored_traces(
        results,
        cases,
        CampaignManifest(1, "sqlite:///:memory:", "1", "campaign", str(path)),
    )
    summary = aggregate_campaign(rescored)
    fields = [
        comparison.passed for result in rescored for comparison in result.parameter_comparisons
    ]
    patterns = {case.case_id: case.pattern for case in summary.cases}

    assert sum(result.trajectory_passed for result in rescored) == 36
    assert sum(result.parameters_passed for result in rescored) == 34
    assert (sum(fields), len(fields)) == (49, 51)
    assert sum(result.goal_completion_passed for result in rescored) == 35
    assert summary.suite.overall_successful_runs == 33
    assert summary.suite.mixed_result_scenarios == 2
    assert patterns["chat_simple"] == "2/3"
    assert patterns["write_with_memory"] == "1/3"
    assert patterns["retrieve_exact_draft"] == "3/3"
