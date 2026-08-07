"""Isolated three-run agent evaluation over stored MLflow traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from editorial_team.evaluation.agent_cases import (
    AgentEvaluationCase,
    ParameterExpectation,
)
from editorial_team.evaluation.generation_judges import GenerationMetric, StructuredGenerationJudge
from editorial_team.evaluation.retrieval_metrics import metrics_at_k
from editorial_team.evaluation.trace_adapters import (
    EvaluationToolCall,
    GenerationReference,
    RetrievalReference,
    trace_to_generation_judge_input,
    trace_to_retrieval_scorer_input,
    trace_to_tool_calls,
)

EVALUATION_AGENT_TEMPERATURE = 0.2
RUNS_PER_CASE = 3


@dataclass(frozen=True)
class RunIdentity:
    case_id: str
    run_number: int
    conversation_id: str
    thread_id: str
    request_origin: str = "batch"


@dataclass(frozen=True)
class AgentInvocation:
    trace: Any
    trace_id: str
    final_response: str
    outcome_facts: Mapping[str, bool]
    retrieval_reference: RetrievalReference | None = None
    generation_reference: GenerationReference | None = None


class AgentRunExecutor(Protocol):
    def execute(
        self,
        case: AgentEvaluationCase,
        identity: RunIdentity,
        *,
        agent_temperature: float,
    ) -> AgentInvocation: ...


class AgentRunFailure(RuntimeError):
    """Sanitized failed invocation with its persisted trace identity when available."""

    def __init__(self, trace_id: str, public_error: str) -> None:
        super().__init__(public_error)
        self.trace_id = trace_id
        self.public_error = public_error


@dataclass(frozen=True)
class ParameterComparison:
    call_index: int
    field: str
    passed: bool
    expected: object
    observed: object


@dataclass(frozen=True)
class AgentRunResult:
    case_id: str
    run_number: int
    conversation_id: str
    thread_id: str
    request_origin: str
    agent_temperature: float
    trace_id: str
    tool_trajectory: tuple[str, ...]
    accepted_trajectories: tuple[tuple[str, ...], ...]
    trajectory_passed: bool
    parameter_comparisons: tuple[ParameterComparison, ...]
    parameters_passed: bool
    goal_completion_passed: bool
    retrieval_scores: Mapping[str, float] | None
    generation_scores: Mapping[str, float] | None
    error: str | None


def run_identities(
    cases: Sequence[AgentEvaluationCase], *, runs_per_case: int = RUNS_PER_CASE
) -> tuple[RunIdentity, ...]:
    if runs_per_case != RUNS_PER_CASE:
        raise ValueError("HW3 agent evaluation requires exactly three runs per case")
    return tuple(
        _run_identity(case.case_id, run_number)
        for case in cases
        for run_number in range(1, RUNS_PER_CASE + 1)
    )


def _run_identity(case_id: str, run_number: int) -> RunIdentity:
    conversation_id = f"eval-{case_id.replace('_', '-')}-r{run_number}"
    return RunIdentity(
        case_id,
        run_number,
        conversation_id,
        f"editorial:v1:{conversation_id}",
    )


def run_agent_evaluation(
    cases: Sequence[AgentEvaluationCase],
    executor: AgentRunExecutor,
    *,
    output_path: Path | None = None,
    agent_temperature: float = EVALUATION_AGENT_TEMPERATURE,
    generation_judge: StructuredGenerationJudge | None = None,
) -> tuple[AgentRunResult, ...]:
    """Execute independent calls and evaluate only their persisted trace representation."""

    if not 0 < agent_temperature <= 2:
        raise ValueError("agent_temperature must be greater than zero and at most two")
    case_map = {case.case_id: case for case in cases}
    identities = run_identities(cases)
    results: list[AgentRunResult] = []
    for identity in identities:
        case = case_map[identity.case_id]
        try:
            invocation = executor.execute(case, identity, agent_temperature=agent_temperature)
            calls = trace_to_tool_calls(invocation.trace)
            trajectory = tuple(call.tool for call in calls)
            comparisons = compare_parameters(calls, case.parameter_expectations)
            retrieval_scores = _retrieval_scores(case, invocation)
            generation_scores = _generation_scores(case, invocation, generation_judge)
            result = AgentRunResult(
                case_id=case.case_id,
                run_number=identity.run_number,
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                request_origin=identity.request_origin,
                agent_temperature=agent_temperature,
                trace_id=invocation.trace_id,
                tool_trajectory=trajectory,
                accepted_trajectories=case.accepted_trajectories,
                trajectory_passed=trajectory in case.accepted_trajectories,
                parameter_comparisons=comparisons,
                parameters_passed=all(item.passed for item in comparisons),
                goal_completion_passed=evaluate_goal(case, invocation),
                retrieval_scores=retrieval_scores,
                generation_scores=generation_scores,
                error=None,
            )
        except Exception as exc:
            trace_id = exc.trace_id if isinstance(exc, AgentRunFailure) else ""
            error = (
                exc.public_error
                if isinstance(exc, AgentRunFailure)
                else f"{type(exc).__name__}: {exc}"
            )
            result = AgentRunResult(
                case_id=case.case_id,
                run_number=identity.run_number,
                conversation_id=identity.conversation_id,
                thread_id=identity.thread_id,
                request_origin=identity.request_origin,
                agent_temperature=agent_temperature,
                trace_id=trace_id,
                tool_trajectory=(),
                accepted_trajectories=case.accepted_trajectories,
                trajectory_passed=False,
                parameter_comparisons=(),
                parameters_passed=False,
                goal_completion_passed=False,
                retrieval_scores=None,
                generation_scores=None,
                error=error,
            )
        results.append(result)
        if output_path is not None:
            write_results(output_path, results)
    return tuple(results)


def compare_parameters(
    calls: Sequence[EvaluationToolCall],
    expectations: Sequence[ParameterExpectation],
) -> tuple[ParameterComparison, ...]:
    output: list[ParameterComparison] = []
    for expectation in expectations:
        observed: object = None
        if 0 <= expectation.call_index < len(calls):
            observed = calls[expectation.call_index].arguments.get(expectation.field)
        passed = _parameter_matches(expectation, observed)
        output.append(
            ParameterComparison(
                expectation.call_index,
                expectation.field,
                passed,
                expectation.expected,
                observed,
            )
        )
    return tuple(output)


def evaluate_goal(case: AgentEvaluationCase, invocation: AgentInvocation) -> bool:
    """Evaluate declared outcome criteria independently of trajectory correctness."""

    expectation = case.outcome
    response = invocation.final_response.strip()
    normalized = response.casefold()
    if expectation.require_response and not response:
        return False
    if any(term.casefold() not in normalized for term in expectation.required_response_terms):
        return False
    if any(term.casefold() in normalized for term in expectation.forbidden_response_terms):
        return False
    return all(invocation.outcome_facts.get(fact) is True for fact in expectation.required_facts)


def write_results(path: Path, results: Sequence[AgentRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parameter_matches(expectation: ParameterExpectation, observed: object) -> bool:
    if expectation.mode == "equals":
        return observed == expectation.expected
    if expectation.mode == "non_empty":
        return isinstance(observed, str) and bool(observed.strip())
    if expectation.mode == "timestamp_equals":
        if not isinstance(observed, str) or not isinstance(expectation.expected, str):
            return False
        try:
            return datetime.fromisoformat(
                observed.replace("Z", "+00:00")
            ) == datetime.fromisoformat(expectation.expected.replace("Z", "+00:00"))
        except ValueError:
            return False
    if expectation.mode == "contains_terms":
        if not isinstance(observed, str) or not isinstance(expectation.expected, tuple):
            return False
        normalized = " ".join(observed.casefold().split())
        return all(
            isinstance(term, str) and term.casefold() in normalized for term in expectation.expected
        )
    return False


def _retrieval_scores(
    case: AgentEvaluationCase, invocation: AgentInvocation
) -> Mapping[str, float] | None:
    if not case.score_retrieval:
        return None
    if invocation.retrieval_reference is None:
        raise ValueError("retrieval-scored case is missing its bounded reference")
    adapted = trace_to_retrieval_scorer_input(
        invocation.trace, {case.case_id: invocation.retrieval_reference}
    )
    score = metrics_at_k(adapted.predictions, adapted.golden, 5)
    return {
        "precision_at_5": score.precision,
        "recall_at_5": score.recall,
        "mrr_at_5": score.mrr,
    }


def _generation_scores(
    case: AgentEvaluationCase,
    invocation: AgentInvocation,
    judge: StructuredGenerationJudge | None,
) -> Mapping[str, float] | None:
    if not case.score_generation:
        return None
    if invocation.generation_reference is None:
        raise ValueError("generation-scored case is missing its bounded reference")
    adapted = trace_to_generation_judge_input(
        invocation.trace, {case.case_id: invocation.generation_reference}
    )
    if judge is None:
        return {"adapter_exercised": 1.0}
    output: dict[str, float] = {}
    for metric in GenerationMetric:
        score = judge.judge(
            metric,
            query=adapted.query,
            candidate_answer=adapted.candidate_answer,
            golden_answer=adapted.golden_answer,
            retrieved_contexts=adapted.retrieved_contexts,
            golden_contexts=adapted.golden_contexts,
        )
        output[metric.value] = score.score
    return output
