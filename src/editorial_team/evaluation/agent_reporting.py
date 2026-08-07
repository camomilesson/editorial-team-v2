"""Canonical Stage 5 aggregation and trace feedback for agent evaluations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mlflow

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    ParagraphChunker,
    artifact_id_for,
    content_sha256,
)
from editorial_team.evaluation.agent_cases import AgentEvaluationCase
from editorial_team.evaluation.agent_harness import (
    RUNS_PER_CASE,
    AgentRunResult,
    ParameterComparison,
    compare_parameters,
)
from editorial_team.evaluation.generation_judges import GenerationMetric, StructuredGenerationJudge
from editorial_team.evaluation.generation_models import GenerationContext
from editorial_team.evaluation.retrieval_metrics import metrics_at_k
from editorial_team.evaluation.trace_adapters import (
    GenerationReference,
    RetrievalReference,
    trace_to_generation_judge_input,
    trace_to_retrieval_scorer_input,
    trace_to_tool_calls,
)
from editorial_team.safety import score_trace_safety

PART1_FEEDBACK_NAMES = {
    "tool_selection": "agent.tool_selection_accuracy",
    "tool_parameters": "agent.tool_parameter_accuracy",
    "goal_completion": "agent.goal_completion",
    "overall_pass": "agent.overall_pass",
}
RETRIEVAL_FEEDBACK_NAMES = {
    "mrr_at_5": "retrieval.mrr_at_5",
    "precision_at_5": "retrieval.precision_at_5",
    "recall_at_5": "retrieval.recall_at_5",
}
GENERATION_FEEDBACK_NAMES = {
    "faithfulness": "generation.faithfulness",
    "answer_relevance": "generation.answer_relevance",
    "context_precision": "generation.context_precision",
    "context_recall": "generation.context_recall",
}
SAFETY_FEEDBACK_NAMES = {
    "threat_detected": "safety.threat_detected",
    "defense_effective": "safety.defense_effective",
    "unsafe_behavior": "safety.unsafe_behavior",
}


@dataclass(frozen=True)
class TraceLocation:
    tracking_uri: str
    experiment_id: str
    experiment_name: str

    def __post_init__(self) -> None:
        _validate_tracking_identity(self.tracking_uri, self.experiment_id)


@dataclass(frozen=True)
class CampaignManifest:
    schema_version: int
    tracking_uri: str
    experiment_id: str
    experiment_name: str
    raw_results_path: str
    trace_locations: Mapping[str, TraceLocation] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported campaign manifest schema")
        _validate_tracking_identity(self.tracking_uri, self.experiment_id)
        if self.trace_locations is not None and not all(
            isinstance(trace_id, str) and trace_id.strip() and isinstance(location, TraceLocation)
            for trace_id, location in self.trace_locations.items()
        ):
            raise ValueError("campaign trace locations are malformed")

    def location_for(self, trace_id: str) -> TraceLocation:
        if self.trace_locations is not None:
            location = self.trace_locations.get(trace_id)
            if location is None:
                raise RuntimeError(f"trace {trace_id} has no campaign tracking location")
            return location
        return TraceLocation(self.tracking_uri, self.experiment_id, self.experiment_name)


@dataclass(frozen=True)
class CanonicalRunResult:
    case_id: str
    run_number: int
    trace_id: str
    tool_trajectory: tuple[str, ...]
    trajectory_passed: bool
    parameters_passed: bool
    goal_completion_passed: bool
    overall_passed: bool
    retrieval_scores: Mapping[str, float] | None
    generation_scores: Mapping[str, float] | None
    error: str | None


@dataclass(frozen=True)
class CaseReliability:
    case_id: str
    pass_count: int
    run_count: int
    pattern: str
    success_rate: float
    pass_at_3: int
    pass_power_3: int
    tool_selection_accuracy: float
    parameter_accuracy: float
    goal_completion_rate: float


@dataclass(frozen=True)
class SuiteAggregate:
    total_scenarios: int
    total_runs: int
    overall_successful_runs: int
    overall_success_rate: float
    tool_selection_accuracy: float
    run_level_parameter_accuracy: float
    field_level_parameter_accuracy: float | None
    goal_completion_rate: float
    mixed_result_scenarios: int


@dataclass(frozen=True)
class CampaignSummary:
    schema_version: int
    runs: tuple[CanonicalRunResult, ...]
    cases: tuple[CaseReliability, ...]
    suite: SuiteAggregate


@dataclass(frozen=True)
class SafetyFeedbackReport:
    evaluated: int
    unevaluable: int
    flagged: int
    assessments_logged: int


def overall_pass(result: AgentRunResult) -> bool:
    return (
        result.trajectory_passed
        and result.parameters_passed
        and result.goal_completion_passed
        and result.error is None
    )


def aggregate_campaign(results: Sequence[AgentRunResult]) -> CampaignSummary:
    """Aggregate a complete campaign while keeping tool, parameter, and goal metrics separate."""

    grouped: dict[str, list[AgentRunResult]] = defaultdict(list)
    for result in results:
        grouped[result.case_id].append(result)
    if not grouped:
        raise ValueError("campaign contains no agent results")
    for case_id, case_results in grouped.items():
        run_numbers = sorted(result.run_number for result in case_results)
        if run_numbers != list(range(1, RUNS_PER_CASE + 1)):
            raise ValueError(f"{case_id} must contain exactly runs 1, 2, and 3")

    ordered = tuple(sorted(results, key=lambda item: (item.case_id, item.run_number)))
    canonical_runs = tuple(
        CanonicalRunResult(
            result.case_id,
            result.run_number,
            result.trace_id,
            result.tool_trajectory,
            result.trajectory_passed,
            result.parameters_passed,
            result.goal_completion_passed,
            overall_pass(result),
            result.retrieval_scores,
            result.generation_scores,
            result.error,
        )
        for result in ordered
    )
    cases: list[CaseReliability] = []
    for case_id in sorted(grouped):
        case_results = sorted(grouped[case_id], key=lambda item: item.run_number)
        pass_count = sum(overall_pass(item) for item in case_results)
        cases.append(
            CaseReliability(
                case_id=case_id,
                pass_count=pass_count,
                run_count=RUNS_PER_CASE,
                pattern=f"{pass_count}/{RUNS_PER_CASE}",
                success_rate=pass_count / RUNS_PER_CASE,
                pass_at_3=int(pass_count > 0),
                pass_power_3=int(pass_count == RUNS_PER_CASE),
                tool_selection_accuracy=_mean(item.trajectory_passed for item in case_results),
                parameter_accuracy=_mean(item.parameters_passed for item in case_results),
                goal_completion_rate=_mean(item.goal_completion_passed for item in case_results),
            )
        )

    field_values = [
        comparison.passed for result in ordered for comparison in result.parameter_comparisons
    ]
    successful = sum(overall_pass(item) for item in ordered)
    suite = SuiteAggregate(
        total_scenarios=len(cases),
        total_runs=len(ordered),
        overall_successful_runs=successful,
        overall_success_rate=successful / len(ordered),
        tool_selection_accuracy=_mean(item.trajectory_passed for item in ordered),
        run_level_parameter_accuracy=_mean(item.parameters_passed for item in ordered),
        field_level_parameter_accuracy=_mean(field_values) if field_values else None,
        goal_completion_rate=_mean(item.goal_completion_passed for item in ordered),
        mixed_result_scenarios=sum(0 < item.pass_count < RUNS_PER_CASE for item in cases),
    )
    return CampaignSummary(1, canonical_runs, tuple(cases), suite)


def load_run_results(path: Path) -> tuple[AgentRunResult, ...]:
    """Load the Stage 4 list format without requiring a manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("raw agent results must be a JSON list")
    output: list[AgentRunResult] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("raw agent result is malformed")
        value = dict(item)
        value["tool_trajectory"] = tuple(value["tool_trajectory"])
        value["accepted_trajectories"] = tuple(
            tuple(trajectory) for trajectory in value["accepted_trajectories"]
        )
        value["parameter_comparisons"] = tuple(
            ParameterComparison(**comparison) for comparison in value["parameter_comparisons"]
        )
        output.append(AgentRunResult(**value))
    return tuple(output)


def write_campaign_summary(path: Path, summary: CampaignSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_campaign_manifest(path: Path, manifest: CampaignManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_campaign_manifest(path: Path) -> CampaignManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("campaign manifest is malformed")
    locations = payload.get("trace_locations")
    if isinstance(locations, dict):
        payload = dict(payload)
        payload["trace_locations"] = {
            trace_id: TraceLocation(**location) for trace_id, location in locations.items()
        }
    return CampaignManifest(**payload)


def log_campaign_feedback(
    results: Sequence[AgentRunResult],
    manifest: CampaignManifest,
    *,
    get_trace: Callable[..., Any] = mlflow.get_trace,
    log_feedback: Callable[..., Any] = mlflow.log_feedback,
    override_feedback: Callable[..., Any] = mlflow.override_feedback,
) -> int:
    """Attach bounded numeric feedback after validating every trace in its explicit store."""

    logged = 0
    for result in results:
        if not result.trace_id:
            continue
        location = manifest.location_for(result.trace_id)
        with _tracking_store(location.tracking_uri):
            trace = get_trace(result.trace_id, flush=True)
            if trace is None:
                raise RuntimeError(
                    f"trace {result.trace_id} is missing from campaign tracking store"
                )
            trace_experiment = getattr(trace.info, "experiment_id", None)
            if trace_experiment is not None and str(trace_experiment) != location.experiment_id:
                raise RuntimeError(f"trace {result.trace_id} belongs to a different experiment")
            metrics = _feedback_values(result)
            for name, value in metrics.items():
                _upsert_feedback(
                    trace,
                    trace_id=result.trace_id,
                    name=name,
                    value=value,
                    metadata={"case_id": result.case_id, "run_number": result.run_number},
                    log_feedback=log_feedback,
                    override_feedback=override_feedback,
                )
                logged += 1
    return logged


def log_campaign_safety_feedback(
    results: Sequence[AgentRunResult],
    manifest: CampaignManifest,
    *,
    get_trace: Callable[..., Any] = mlflow.get_trace,
    log_feedback: Callable[..., Any] = mlflow.log_feedback,
) -> SafetyFeedbackReport:
    """Score stored traces purely and log only bounded numeric safety feedback."""

    evaluated = unevaluable = flagged = logged = 0
    for result in results:
        if not result.trace_id:
            unevaluable += 1
            continue
        location = manifest.location_for(result.trace_id)
        with _tracking_store(location.tracking_uri):
            trace = get_trace(result.trace_id, flush=True)
            if trace is None:
                raise RuntimeError(
                    f"trace {result.trace_id} is missing from campaign tracking store"
                )
            score = score_trace_safety(trace)
            if not score.evaluable:
                unevaluable += 1
                continue
            evaluated += 1
            flagged += int(score.flagged)
            values = {
                SAFETY_FEEDBACK_NAMES["threat_detected"]: score.threat_detected,
                SAFETY_FEEDBACK_NAMES["defense_effective"]: score.defense_effective,
                SAFETY_FEEDBACK_NAMES["unsafe_behavior"]: score.unsafe_behavior,
            }
            for name, value in values.items():
                log_feedback(
                    trace_id=result.trace_id,
                    name=name,
                    value=value,
                    metadata={"case_id": result.case_id, "run_number": result.run_number},
                )
                logged += 1
    return SafetyFeedbackReport(evaluated, unevaluable, flagged, logged)


def replace_scores(
    result: AgentRunResult,
    *,
    retrieval_scores: Mapping[str, float] | None = None,
    generation_scores: Mapping[str, float] | None = None,
) -> AgentRunResult:
    """Return a rescored stored run without mutating its observed Stage 4 result."""

    return replace(
        result,
        retrieval_scores=(
            result.retrieval_scores if retrieval_scores is None else retrieval_scores
        ),
        generation_scores=(
            result.generation_scores if generation_scores is None else generation_scores
        ),
    )


def rescore_part1_from_stored_traces(
    results: Sequence[AgentRunResult],
    cases: Sequence[AgentEvaluationCase],
    manifest: CampaignManifest,
) -> tuple[AgentRunResult, ...]:
    """Correct Part 1 values from frozen traces without agent or scorer-model execution."""

    case_map = {case.case_id: case for case in cases}
    output: list[AgentRunResult] = []
    for result in results:
        case = case_map.get(result.case_id)
        if case is None:
            raise ValueError(f"no declared case exists for {result.case_id}")
        if not result.trace_id:
            output.append(result)
            continue
        location = manifest.location_for(result.trace_id)
        with _tracking_store(location.tracking_uri):
            trace = mlflow.get_trace(result.trace_id, flush=True)
        if trace is None:
            raise RuntimeError(f"trace {result.trace_id} is missing from campaign tracking store")
        calls = trace_to_tool_calls(trace)
        trajectory = tuple(call.tool for call in calls)
        comparisons = compare_parameters(calls, case.parameter_expectations)
        goal = result.goal_completion_passed
        if result.error is not None:
            goal = False
        elif case.case_id == "retrieve_exact_draft":
            goal = _frozen_exact_draft_goal(trace, case, comparisons)
        output.append(
            replace(
                result,
                tool_trajectory=trajectory,
                trajectory_passed=trajectory in case.accepted_trajectories,
                parameter_comparisons=comparisons,
                parameters_passed=all(item.passed for item in comparisons),
                goal_completion_passed=goal,
            )
        )
    return tuple(output)


def rescore_stored_traces(
    results: Sequence[AgentRunResult],
    cases: Sequence[AgentEvaluationCase],
    manifest: CampaignManifest,
    *,
    generation_judge: StructuredGenerationJudge | None = None,
    case_ids: frozenset[str] | None = None,
) -> tuple[AgentRunResult, ...]:
    """Recompute HW2 scores from stored traces and current references without agent execution."""

    case_map = {case.case_id: case for case in cases}
    output: list[AgentRunResult] = []
    for result in results:
        case = case_map.get(result.case_id)
        if case is None:
            raise ValueError(f"no declared case exists for {result.case_id}")
        if case_ids is not None and result.case_id not in case_ids:
            output.append(result)
            continue
        if not result.trace_id or result.error is not None:
            output.append(result)
            continue
        location = manifest.location_for(result.trace_id)
        with _tracking_store(location.tracking_uri):
            trace = mlflow.get_trace(result.trace_id, flush=True)
            if trace is None:
                raise RuntimeError(
                    f"trace {result.trace_id} is missing from campaign tracking store"
                )
            trace_experiment = getattr(trace.info, "experiment_id", None)
            if trace_experiment is not None and str(trace_experiment) != location.experiment_id:
                raise RuntimeError(f"trace {result.trace_id} belongs to a different experiment")
            retrieval_reference, generation_reference = _case_references(case)
            retrieval_scores = result.retrieval_scores
            generation_scores = result.generation_scores
            if case.score_retrieval:
                if retrieval_reference is None:
                    raise ValueError("retrieval-scored case has no current reference")
                adapted = trace_to_retrieval_scorer_input(
                    trace, {case.case_id: retrieval_reference}
                )
                score = metrics_at_k(adapted.predictions, adapted.golden, 5)
                retrieval_scores = {
                    "precision_at_5": score.precision,
                    "recall_at_5": score.recall,
                    "mrr_at_5": score.mrr,
                }
            if case.score_generation and generation_judge is not None:
                if generation_reference is None:
                    raise ValueError("generation-scored case has no current reference")
                adapted = trace_to_generation_judge_input(
                    trace, {case.case_id: generation_reference}
                )
                generation_scores = {
                    metric.value: generation_judge.judge(
                        metric,
                        query=adapted.query,
                        candidate_answer=adapted.candidate_answer,
                        golden_answer=adapted.golden_answer,
                        retrieved_contexts=adapted.retrieved_contexts,
                        golden_contexts=adapted.golden_contexts,
                    ).score
                    for metric in GenerationMetric
                }
            output.append(
                replace_scores(
                    result,
                    retrieval_scores=retrieval_scores,
                    generation_scores=generation_scores,
                )
            )
    return tuple(output)


def _feedback_values(result: AgentRunResult) -> dict[str, float]:
    values = {
        PART1_FEEDBACK_NAMES["tool_selection"]: float(result.trajectory_passed),
        PART1_FEEDBACK_NAMES["tool_parameters"]: float(result.parameters_passed),
        PART1_FEEDBACK_NAMES["goal_completion"]: float(result.goal_completion_passed),
        PART1_FEEDBACK_NAMES["overall_pass"]: float(overall_pass(result)),
    }
    for scores, names in (
        (result.retrieval_scores, RETRIEVAL_FEEDBACK_NAMES),
        (result.generation_scores, GENERATION_FEEDBACK_NAMES),
    ):
        if scores is None:
            continue
        for key, name in names.items():
            value = scores.get(key)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"feedback score {key} is invalid")
                values[name] = float(value)
    return values


def _frozen_exact_draft_goal(
    trace: Any,
    case: AgentEvaluationCase,
    comparisons: Sequence[ParameterComparison],
) -> bool:
    if (
        len(case.setup.artifacts) != 1
        or not comparisons
        or not all(item.passed for item in comparisons)
    ):
        return False
    roots = [span for span in trace.data.spans if span.parent_id is None]
    if len(roots) != 1:
        return False
    candidate = roots[0].attributes.get("evaluation.candidate_answer")
    if not isinstance(candidate, str) or "✍️ Writer\n\n" not in candidate:
        return False
    writer = candidate.split("✍️ Writer\n\n", 1)[1].split("\n\n🔍 Critic", 1)[0].strip()
    source = case.setup.artifacts[0].content.strip()
    return bool(writer and writer != source and "Verdict: PASS" in candidate)


def _upsert_feedback(
    trace: Any,
    *,
    trace_id: str,
    name: str,
    value: float,
    metadata: dict[str, object],
    log_feedback: Callable[..., Any],
    override_feedback: Callable[..., Any],
) -> None:
    assessments = getattr(trace.info, "assessments", ()) or ()
    existing = next(
        (
            assessment
            for assessment in reversed(assessments)
            if assessment.name == name and assessment.valid is not False
        ),
        None,
    )
    if existing is None:
        log_feedback(trace_id=trace_id, name=name, value=value, metadata=metadata)
    else:
        override_feedback(
            trace_id=trace_id,
            assessment_id=existing.assessment_id,
            value=value,
            metadata=metadata,
        )


def _case_references(
    case: AgentEvaluationCase,
) -> tuple[RetrievalReference | None, GenerationReference | None]:
    chunks_by_fixture: dict[str, tuple[Any, ...]] = {}
    chunker = ParagraphChunker()
    for fixture in case.setup.artifacts:
        artifact = EditorialArtifact(
            artifact_id=artifact_id_for(fixture.task_id, ArtifactProducer.WRITER),
            task_id=fixture.task_id,
            producer=ArtifactProducer.WRITER,
            created_at=fixture.created_at,
            conversation_id="eval-reference",
            user_request=fixture.user_request,
            content=fixture.content,
            content_sha256=content_sha256(fixture.content),
        )
        chunks_by_fixture[fixture.fixture_id] = chunker.chunk(artifact)
    golden_chunks = tuple(
        chunk
        for fixture_id in case.golden_fixture_ids
        for chunk in chunks_by_fixture.get(fixture_id, ())
    )
    retrieval = (
        RetrievalReference(case.case_id, frozenset(chunk.chunk_id for chunk in golden_chunks))
        if case.score_retrieval
        else None
    )
    generation = (
        GenerationReference(
            case.case_id,
            case.generation_golden_answer or "",
            tuple(
                GenerationContext(chunk.chunk_id, chunk.artifact_id, chunk.content)
                for chunk in golden_chunks
            ),
        )
        if case.score_generation
        else None
    )
    if case.score_retrieval and not golden_chunks:
        raise ValueError("scored case has no current golden contexts")
    if case.score_generation and not case.generation_golden_answer:
        raise ValueError("generation-scored case has no current golden answer")
    return retrieval, generation


@contextmanager
def _tracking_store(tracking_uri: str) -> Iterator[None]:
    if not isinstance(tracking_uri, str) or not tracking_uri.strip():
        raise ValueError("campaign tracking URI is missing")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def _validate_tracking_identity(tracking_uri: str, experiment_id: str) -> None:
    if not isinstance(tracking_uri, str) or not isinstance(experiment_id, str):
        raise ValueError("campaign tracking identity is incomplete")
    parsed = urlsplit(tracking_uri)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("campaign tracking URI must not embed credentials")
    if not tracking_uri.strip() or not experiment_id.strip():
        raise ValueError("campaign tracking identity is incomplete")


def _mean(values: Sequence[bool] | list[bool] | Any) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("metric denominator must not be empty")
    return sum(bool(value) for value in materialized) / len(materialized)
