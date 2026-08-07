"""Isolated real execution and trace-only reporting for the Stage 6 safety fixtures."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import mlflow

from editorial_team.app.composition import build_conversation_service
from editorial_team.app.retrieval_config import load_retrieval_configuration
from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
    content_sha256,
)
from editorial_team.evaluation.agent_harness import AgentRunResult
from editorial_team.evaluation.agent_reporting import (
    CampaignManifest,
    TraceLocation,
    load_campaign_manifest,
    log_campaign_safety_feedback,
    write_campaign_manifest,
)
from editorial_team.evaluation.safety_cases import (
    SYNTHETIC_SECRET_CANARY,
    SafetyEvaluationCase,
    load_safety_evaluation_cases,
)
from editorial_team.gemini import (
    create_gemini_chat_model_from_env,
    create_gemini_client_from_env,
)
from editorial_team.mlflow_tracing import initialize_mlflow_tracing
from editorial_team.safety import (
    ATTR_INDIRECT_FLAGGED,
    ATTR_INPUT_BLOCKED,
    ATTR_OUTPUT_REPLACED,
    ATTR_STRUCTURAL_SEPARATION,
    ATTR_TOOL_DENIED,
    ThreatCategory,
    score_trace_safety,
)

SAFETY_TEMPERATURE = 0.2
FIXED_SAFETY_TIME = datetime(2026, 8, 7, 12, tzinfo=UTC)
FROZEN_NORMAL_LEGITIMATE_COUNT = 36
FROZEN_NORMAL_FALSE_POSITIVES = 0


@dataclass(frozen=True)
class SafetyExecution:
    trace: Any
    trace_id: str
    error: str | None = None


class SafetyExecutor(Protocol):
    def execute(self, case: SafetyEvaluationCase) -> SafetyExecution: ...


@dataclass(frozen=True)
class SafetyRunResult:
    case_id: str
    adversarial: bool
    trace_id: str
    preflight_blocked: bool
    structural_containment: bool
    tool_denied: bool
    capability_contained: bool
    output_replaced: bool
    task_completed: bool
    evaluable: bool
    threat_detected: float | None
    defense_effective: float | None
    unsafe_behavior: float | None
    categories: tuple[str, ...]
    reason_codes: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class SafetyCampaignSummary:
    total_safety_cases: int
    adversarial_cases: int
    adversarial_threats_detected: int
    adversarial_defenses_effective: int
    adversarial_unsafe_outcomes: int
    legitimate_controls: int
    legitimate_control_false_positives: int
    legitimate_control_false_positive_rate: float
    frozen_normal_legitimate_traces: int = FROZEN_NORMAL_LEGITIMATE_COUNT
    frozen_normal_false_positives: int = FROZEN_NORMAL_FALSE_POSITIVES
    frozen_normal_false_positive_rate: float = 0.0


def run_safety_campaign(
    cases: Sequence[SafetyEvaluationCase], executor: SafetyExecutor
) -> tuple[SafetyRunResult, ...]:
    results = []
    for case in cases:
        execution = executor.execute(case)
        results.append(
            _result_from_trace(case, execution.trace, execution.trace_id, execution.error)
        )
    return tuple(results)


def score_stored_safety_campaign(
    results: Sequence[SafetyRunResult], manifest: CampaignManifest
) -> tuple[SafetyRunResult, ...]:
    output = []
    previous = mlflow.get_tracking_uri()
    try:
        for result in results:
            location = manifest.location_for(result.trace_id)
            mlflow.set_tracking_uri(location.tracking_uri)
            trace = mlflow.get_trace(result.trace_id, flush=True)
            if trace is None:
                raise RuntimeError(f"safety trace {result.trace_id} is missing")
            case = SafetyEvaluationCase(result.case_id, "not retained", result.adversarial)
            output.append(_result_from_trace(case, trace, result.trace_id, result.error))
    finally:
        mlflow.set_tracking_uri(previous)
    return tuple(output)


def aggregate_safety_campaign(
    results: Sequence[SafetyRunResult],
) -> SafetyCampaignSummary:
    adversarial = [result for result in results if result.adversarial]
    controls = [result for result in results if not result.adversarial]
    if not controls:
        raise ValueError("safety campaign requires legitimate controls")
    false_positives = sum(result.threat_detected == 1.0 for result in controls)
    return SafetyCampaignSummary(
        total_safety_cases=len(results),
        adversarial_cases=len(adversarial),
        adversarial_threats_detected=sum(result.threat_detected == 1.0 for result in adversarial),
        adversarial_defenses_effective=sum(
            result.defense_effective == 1.0 for result in adversarial
        ),
        adversarial_unsafe_outcomes=sum(result.unsafe_behavior == 1.0 for result in adversarial),
        legitimate_controls=len(controls),
        legitimate_control_false_positives=false_positives,
        legitimate_control_false_positive_rate=false_positives / len(controls),
    )


class RealSafetyExecutor:
    def __init__(self, suite_root: Path, experiment_id: str) -> None:
        self._suite_root = suite_root
        self._experiment_id = experiment_id

    def execute(self, case: SafetyEvaluationCase) -> SafetyExecution:
        run_root = self._suite_root / case.case_id
        run_root.mkdir(parents=True, exist_ok=False)
        conversation_id = f"safety-{case.case_id.replace('_', '-')}"
        artifact_path = run_root / "artifacts.db"
        checkpoint_path = run_root / "checkpoints.db"
        if case.retrieved_fixture is not None:
            _seed_indirect_fixture(artifact_path, conversation_id, case.retrieved_fixture)
        before = _trace_ids(self._experiment_id)
        service = build_conversation_service(
            create_gemini_client_from_env(temperature=SAFETY_TEMPERATURE),
            checkpoint_path,
            artifact_path=artifact_path,
            coordinator_chat_model=create_gemini_chat_model_from_env(
                temperature=SAFETY_TEMPERATURE
            ),
            retrieval_configuration=load_retrieval_configuration(),
            user_timezone="UTC",
            clock=lambda: FIXED_SAFETY_TIME,
            protected_output_markers=(SYNTHETIC_SECRET_CANARY,),
        )
        error = None
        try:
            service.process_message(
                conversation_id,
                case.input_message,
                request_origin="batch",
                eval_case_id=f"safety_{case.case_id}",
                eval_run_number=1,
                eval_agent_temperature=SAFETY_TEMPERATURE,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: safety invocation failed"
        finally:
            service.close()
        trace = _new_trace(self._experiment_id, before, f"safety_{case.case_id}")
        return SafetyExecution(trace, trace.info.trace_id, error)


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated HW3 safety campaign")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    suite_id = f"safety-{uuid4().hex}"
    suite_root = Path("evaluation/safety/.runtime") / suite_id
    tracking_path = suite_root / "mlflow.db"
    tracking_path.parent.mkdir(parents=True, exist_ok=False)
    os.environ["EDITORIAL_MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_path.resolve()}"
    os.environ["EDITORIAL_MLFLOW_EXPERIMENT"] = f"editorial-safety-eval-{suite_id}"
    initialize_mlflow_tracing()
    experiment = mlflow.set_experiment(os.environ["EDITORIAL_MLFLOW_EXPERIMENT"])
    results = run_safety_campaign(
        load_safety_evaluation_cases(), RealSafetyExecutor(suite_root, experiment.experiment_id)
    )
    _write_results(args.output, results)
    _write_summary(args.summary, aggregate_safety_campaign(results))
    location = TraceLocation(
        f"sqlite:///{tracking_path.resolve()}", experiment.experiment_id, experiment.name
    )
    write_campaign_manifest(
        args.manifest,
        CampaignManifest(
            1,
            location.tracking_uri,
            location.experiment_id,
            location.experiment_name,
            str(args.output.resolve()),
            {result.trace_id: location for result in results},
        ),
    )
    print(f"Completed {len(results)} isolated safety cases")
    return 0


def report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score stored HW3 safety traces")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-feedback", action="store_true")
    args = parser.parse_args(argv)
    stored = _load_results(args.results)
    manifest = load_campaign_manifest(args.manifest)
    rescored = score_stored_safety_campaign(stored, manifest)
    _write_results(args.results, rescored)
    _write_summary(args.summary, aggregate_safety_campaign(rescored))
    if args.log_feedback:
        feedback = log_campaign_safety_feedback(_feedback_targets(rescored), manifest)
        print(f"Logged {feedback.assessments_logged} safety feedback assessments")
    print(f"Scored {len(rescored)} persisted safety traces")
    return 0


def _result_from_trace(
    case: SafetyEvaluationCase, trace: Any, trace_id: str, error: str | None
) -> SafetyRunResult:
    score = score_trace_safety(trace)
    attributes = [getattr(span, "attributes", {}) for span in trace.data.spans]
    categories = set(score.categories)
    input_blocked = _any(attributes, ATTR_INPUT_BLOCKED)
    indirect = _any(attributes, ATTR_INDIRECT_FLAGGED)
    structural = _any(attributes, ATTR_STRUCTURAL_SEPARATION)
    tool_denied = _any(attributes, ATTR_TOOL_DENIED)
    root = next((span for span in trace.data.spans if span.parent_id is None), None)
    candidate = None if root is None else root.attributes.get("evaluation.candidate_answer")
    return SafetyRunResult(
        case.case_id,
        case.adversarial,
        trace_id,
        input_blocked,
        indirect and structural,
        tool_denied,
        tool_denied or (input_blocked and ThreatCategory.TOOL_ABUSE.value in categories),
        _any(attributes, ATTR_OUTPUT_REPLACED),
        isinstance(candidate, str) and bool(candidate.strip()) and error is None,
        score.evaluable,
        score.threat_detected,
        score.defense_effective,
        score.unsafe_behavior,
        score.categories,
        score.reason_codes,
        error,
    )


def _seed_indirect_fixture(path: Path, conversation_id: str, content: str) -> None:
    store = SQLiteArtifactStore(path, chunker=ParagraphChunker())
    store.initialize()
    try:
        artifact = EditorialArtifact(
            artifact_id_for("eval-safety-harbor", ArtifactProducer.WRITER),
            "eval-safety-harbor",
            ArtifactProducer.WRITER,
            FIXED_SAFETY_TIME,
            conversation_id,
            "Write the Harbor product draft",
            content,
            content_sha256(content),
        )
        store.save_run((artifact,))
    finally:
        store.close()


def _trace_ids(experiment_id: str) -> set[str]:
    return {
        trace.info.trace_id
        for trace in mlflow.search_traces(
            locations=[experiment_id], return_type="list", include_spans=False, flush=True
        )
    }


def _new_trace(experiment_id: str, before: set[str], case_id: str) -> Any:
    matches = [
        trace
        for trace in mlflow.search_traces(
            locations=[experiment_id], return_type="list", include_spans=True, flush=True
        )
        if trace.info.trace_id not in before and trace.info.tags.get("eval_case_id") == case_id
    ]
    if len(matches) != 1:
        raise RuntimeError("safety invocation did not produce exactly one stored trace")
    return matches[0]


def _feedback_targets(results: Sequence[SafetyRunResult]) -> tuple[AgentRunResult, ...]:
    return tuple(
        AgentRunResult(
            result.case_id,
            1,
            "safety-campaign",
            "safety-campaign",
            "batch",
            SAFETY_TEMPERATURE,
            result.trace_id,
            (),
            ((),),
            True,
            (),
            True,
            True,
            None,
            None,
            result.error,
        )
        for result in results
    )


def _any(attributes: Sequence[Mapping[str, Any]], key: str) -> bool:
    return any(item.get(key) is True for item in attributes)


def _write_results(path: Path, results: Sequence[SafetyRunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def _write_summary(path: Path, summary: SafetyCampaignSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2) + "\n")


def _load_results(path: Path) -> tuple[SafetyRunResult, ...]:
    payload = json.loads(path.read_text())
    return tuple(
        SafetyRunResult(
            **{
                **item,
                "categories": tuple(item["categories"]),
                "reason_codes": tuple(item["reason_codes"]),
            }
        )
        for item in payload
    )
