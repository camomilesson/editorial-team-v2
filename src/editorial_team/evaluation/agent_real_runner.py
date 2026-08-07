"""Manual real-model executor for the isolated HW3 agent evaluation suite."""

from __future__ import annotations

import argparse
import os
from datetime import UTC
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import mlflow
from mlflow.entities import Trace

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
from editorial_team.conversation import ConversationServiceError
from editorial_team.domain.conversation import ConversationState
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.evaluation.agent_cases import (
    FIXED_NOW,
    ActiveTaskFixture,
    AgentEvaluationCase,
    load_agent_evaluation_cases,
)
from editorial_team.evaluation.agent_harness import (
    AgentInvocation,
    AgentRunExecutor,
    AgentRunFailure,
    RunIdentity,
    run_agent_evaluation,
)
from editorial_team.evaluation.agent_reporting import (
    CampaignManifest,
    TraceLocation,
    write_campaign_manifest,
)
from editorial_team.evaluation.generation_judges import StructuredGenerationJudge
from editorial_team.evaluation.generation_models import GenerationContext
from editorial_team.evaluation.trace_adapters import GenerationReference, RetrievalReference
from editorial_team.gemini import (
    create_gemini_chat_model_from_env,
    create_gemini_client_from_env,
)
from editorial_team.mlflow_tracing import initialize_mlflow_tracing


class RealAgentRunExecutor(AgentRunExecutor):
    """Create fresh SQLite state and real model clients for every repetition."""

    def __init__(self, suite_root: Path, experiment_id: str) -> None:
        self._suite_root = suite_root
        self._experiment_id = experiment_id

    def execute(
        self,
        case: AgentEvaluationCase,
        identity: RunIdentity,
        *,
        agent_temperature: float,
    ) -> AgentInvocation:
        run_root = self._suite_root / case.case_id / f"run-{identity.run_number}"
        run_root.mkdir(parents=True, exist_ok=False)
        checkpoint_path = run_root / "checkpoints.db"
        artifact_path = run_root / "artifacts.db"
        retrieval_reference, generation_reference = _seed_artifacts(
            artifact_path, case, identity.conversation_id
        )
        before = _trace_ids(self._experiment_id)
        model = create_gemini_client_from_env(temperature=agent_temperature)
        chat_model = create_gemini_chat_model_from_env(temperature=agent_temperature)
        service = build_conversation_service(
            model,
            checkpoint_path,
            artifact_path=artifact_path,
            coordinator_chat_model=chat_model,
            retrieval_configuration=load_retrieval_configuration(),
            user_timezone="UTC",
            clock=lambda: FIXED_NOW,
        )
        try:
            _seed_active_state(service, identity, case.setup.active_task)
            try:
                messages, active_before, active_after = _invoke_case(
                    service,
                    case,
                    identity,
                    agent_temperature=agent_temperature,
                )
            except Exception as exc:
                try:
                    trace_id = _new_stored_trace(
                        self._experiment_id, before, case.case_id, identity.run_number
                    ).info.trace_id
                except RuntimeError:
                    trace_id = ""
                public_error = (
                    f"ConversationServiceError: {exc}"
                    if isinstance(exc, ConversationServiceError)
                    else "Evaluation invocation failed"
                )
                raise AgentRunFailure(trace_id, public_error) from None
        finally:
            service.close()
        trace = _new_stored_trace(self._experiment_id, before, case.case_id, identity.run_number)
        response = "\n\n".join(message.content for message in messages)
        facts = {
            "active_revision_applied": _active_revision_applied(active_before, active_after),
            "historical_orbit_selected": _historical_orbit_selected(active_after),
            "ember_not_used_as_source": _ember_not_used_as_source(active_before, active_after),
            "no_match_safe": active_before == active_after,
        }
        return AgentInvocation(
            trace=trace,
            trace_id=trace.info.trace_id,
            final_response=response,
            outcome_facts=MappingProxyType(facts),
            retrieval_reference=retrieval_reference,
            generation_reference=generation_reference,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HW3 agent evaluation suite")
    parser.add_argument("--output", type=Path, default=Path("evaluation/agent/results.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Campaign manifest path (defaults beside the raw result JSON)",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--case", action="append", dest="case_ids", help="Run only this case (repeatable)"
    )
    args = parser.parse_args(argv)
    suite_id = f"suite-{uuid4().hex}"
    suite_root = Path("evaluation/agent/.runtime") / suite_id
    tracking_path = suite_root / "mlflow.db"
    tracking_path.parent.mkdir(parents=True, exist_ok=False)
    os.environ["EDITORIAL_MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_path}"
    os.environ["EDITORIAL_MLFLOW_EXPERIMENT"] = f"editorial-agent-eval-{suite_id}"
    initialize_mlflow_tracing()
    experiment = mlflow.set_experiment(os.environ["EDITORIAL_MLFLOW_EXPERIMENT"])
    manifest_path = args.manifest or args.output.with_name(f"{args.output.stem}.manifest.json")
    executor = RealAgentRunExecutor(suite_root, experiment.experiment_id)
    judge = StructuredGenerationJudge(create_gemini_client_from_env())
    cases = load_agent_evaluation_cases()
    if args.case_ids:
        selected = set(args.case_ids)
        cases = tuple(case for case in cases if case.case_id in selected)
        if {case.case_id for case in cases} != selected:
            parser.error("--case contains an unknown case ID")
    results = run_agent_evaluation(
        cases,
        executor,
        output_path=args.output,
        agent_temperature=args.temperature,
        generation_judge=judge,
    )
    location = TraceLocation(
        f"sqlite:///{tracking_path.resolve()}", experiment.experiment_id, experiment.name
    )
    write_campaign_manifest(
        manifest_path,
        CampaignManifest(
            schema_version=1,
            tracking_uri=location.tracking_uri,
            experiment_id=location.experiment_id,
            experiment_name=location.experiment_name,
            raw_results_path=str(args.output.resolve()),
            trace_locations={result.trace_id: location for result in results if result.trace_id},
        ),
    )
    passed = sum(
        result.trajectory_passed
        and result.parameters_passed
        and result.goal_completion_passed
        and result.error is None
        for result in results
    )
    print(f"Completed {len(results)} runs; {passed} passed all Stage 4 checks")
    print(f"Results: {args.output}")
    return 0


def _seed_artifacts(
    path: Path, case: AgentEvaluationCase, conversation_id: str
) -> tuple[RetrievalReference | None, GenerationReference | None]:
    store = SQLiteArtifactStore(path, chunker=ParagraphChunker())
    store.initialize()
    chunks_by_fixture: dict[str, tuple[object, ...]] = {}
    try:
        for fixture in case.setup.artifacts:
            artifact = EditorialArtifact(
                artifact_id=artifact_id_for(fixture.task_id, ArtifactProducer.WRITER),
                task_id=fixture.task_id,
                producer=ArtifactProducer.WRITER,
                created_at=fixture.created_at.astimezone(UTC),
                conversation_id=conversation_id,
                user_request=fixture.user_request,
                content=fixture.content,
                content_sha256=content_sha256(fixture.content),
            )
            store.save_run((artifact,))
            chunks_by_fixture[fixture.fixture_id] = store.get_chunks(artifact.artifact_id)
    finally:
        store.close()
    golden_chunks = tuple(
        chunk
        for fixture_id in case.golden_fixture_ids
        for chunk in chunks_by_fixture.get(fixture_id, ())
    )
    if (case.score_retrieval or case.score_generation) and not golden_chunks:
        raise ValueError("scored case has no resolved golden fixture chunks")
    if not case.score_retrieval:
        retrieval = None
    else:
        retrieval = RetrievalReference(
            case.case_id, frozenset(chunk.chunk_id for chunk in golden_chunks)
        )
    if not case.score_generation:
        generation = None
    else:
        contexts = tuple(
            GenerationContext(chunk.chunk_id, chunk.artifact_id, chunk.content)
            for chunk in golden_chunks
        )
        if not isinstance(case.generation_golden_answer, str):
            raise ValueError("generation-scored case has no factual golden answer")
        generation = GenerationReference(case.case_id, case.generation_golden_answer, contexts)
    return retrieval, generation


def _trace_ids(experiment_id: str) -> set[str]:
    return {
        trace.info.trace_id
        for trace in mlflow.search_traces(
            locations=[experiment_id], return_type="list", include_spans=True, flush=True
        )
    }


def _new_stored_trace(experiment_id: str, before: set[str], case_id: str, run_number: int) -> Trace:
    traces = mlflow.search_traces(
        locations=[experiment_id], return_type="list", include_spans=True, flush=True
    )
    matches = [
        trace
        for trace in traces
        if trace.info.trace_id not in before
        and trace.info.tags.get("eval_case_id") == case_id
        and any(
            span.parent_id is None and span.attributes.get("evaluation.run_number") == run_number
            for span in trace.data.spans
        )
    ]
    if len(matches) != 1:
        raise RuntimeError("evaluation invocation did not produce exactly one stored trace")
    return Trace.from_json(matches[0].to_json())


def _active_task(service: Any, thread_id: str) -> object:
    """Read durable state for outcome checks only, never for tool inference."""

    runner = service._graph_runner
    snapshot = runner.get_state({"configurable": {"thread_id": thread_id}})
    conversation = snapshot.values.get("conversation")
    return None if conversation is None else conversation.active_task


def _invoke_case(
    service: Any,
    case: AgentEvaluationCase,
    identity: RunIdentity,
    *,
    agent_temperature: float,
) -> tuple[tuple[Any, ...], object, object]:
    active_before = _active_task(service, identity.thread_id)
    messages = service.process_message(
        identity.conversation_id,
        case.input_message,
        request_origin="batch",
        eval_case_id=case.case_id,
        eval_run_number=identity.run_number,
        eval_agent_temperature=agent_temperature,
    )
    active_after = _active_task(service, identity.thread_id)
    return messages, active_before, active_after


def _seed_active_state(
    service: Any, identity: RunIdentity, fixture: ActiveTaskFixture | None
) -> None:
    if fixture is None:
        return
    report = CriticReport(CriticVerdict.PASS, "Deterministic evaluation fixture")
    task = WritingTask(
        fixture.task_id,
        identity.conversation_id,
        WritingBrief(fixture.original_request),
        WritingTaskStatus.REVIEWED,
        FIXED_NOW,
        FIXED_NOW,
        fixture.working_draft,
        report,
    )
    service._graph_runner.update_state(
        {"configurable": {"thread_id": identity.thread_id}},
        {"conversation": ConversationState(identity.conversation_id, active_task=task)},
    )


def _active_revision_applied(before: object, after: object) -> bool:
    return (
        isinstance(before, WritingTask)
        and isinstance(after, WritingTask)
        and before.brief.original_request == after.brief.original_request
        and before.working_draft != after.working_draft
        and len(after.brief.instructions) > len(before.brief.instructions)
    )


def _historical_orbit_selected(after: object) -> bool:
    return (
        isinstance(after, WritingTask)
        and after.brief.original_request == "Write the Orbit launch draft"
        and after.working_draft is not None
    )


def _ember_not_used_as_source(before: object, after: object) -> bool:
    return (
        isinstance(before, WritingTask)
        and isinstance(after, WritingTask)
        and "Ember" in before.brief.original_request
        and after.brief.original_request != before.brief.original_request
        and after.working_draft != before.working_draft
    )


if __name__ == "__main__":
    raise SystemExit(main())
