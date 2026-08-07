from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
from mlflow.entities import Trace

from editorial_team.domain.conversation import Message, MessageRole
from editorial_team.evaluation.generation_judges import GenerationMetric, judge_prompt
from editorial_team.evaluation.generation_models import GenerationContext
from editorial_team.evaluation.retrieval_metrics import metrics_at_k
from editorial_team.evaluation.trace_adapters import (
    EvaluationToolCall,
    GenerationReference,
    RetrievalReference,
    TraceAdapterError,
    trace_to_generation_judge_input,
    trace_to_retrieval_scorer_input,
    trace_to_tool_calls,
)
from editorial_team.mlflow_tracing import (
    agent_invocation_span,
    initialize_mlflow_tracing,
    record_batch_candidate_answer,
    record_retrieval_results,
    record_tool_result,
    retrieval_span,
    tool_execution_span,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)
SEARCH_ARGUMENTS = {
    "query": "Aurora launch",
    "created_from": "2026-01-01T00:00:00+00:00",
    "created_to": None,
    "prefer_recent": True,
    "top_k": 2,
    "rerank": True,
}


@pytest.fixture
def trace_experiment(tmp_path: Path, request: pytest.FixtureRequest) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.set_experiment(f"stage-3-{request.node.name}")
    initialize_mlflow_tracing()
    return experiment.experiment_id


def _stored_trace(experiment_id: str, before: set[str]) -> Trace:
    traces = mlflow.search_traces(
        locations=[experiment_id], return_type="list", include_spans=True, flush=True
    )
    captured = [trace for trace in traces if trace.info.trace_id not in before]
    assert len(captured) == 1
    return Trace.from_json(captured[0].to_json())


def _trace_ids(experiment_id: str) -> set[str]:
    return {
        trace.info.trace_id
        for trace in mlflow.search_traces(
            locations=[experiment_id], return_type="list", include_spans=True, flush=True
        )
    }


def _retrieval_data(order: tuple[str, ...], *, rerank: bool = True) -> tuple[object, object]:
    request = SimpleNamespace(
        query="Aurora launch",
        created_from=datetime(2026, 1, 1, tzinfo=UTC),
        created_to=None,
        prefer_recent=True,
        top_k=2,
        rerank=rerank,
    )

    def candidate(chunk_id: str, rank: int) -> object:
        return SimpleNamespace(
            chunk=SimpleNamespace(chunk=SimpleNamespace(chunk_id=chunk_id)), rank=rank
        )

    results = tuple(
        SimpleNamespace(
            chunk_id=chunk_id,
            artifact_id=f"artifact-{chunk_id[-1]}",
            rank=rank,
            dense_score=0.9 - rank / 10,
            bm25_score=3.0 - rank,
            rrf_score=0.04 - rank / 100,
            rerank_score=(0.95 - rank / 10) if rerank else None,
            excerpt=f"context for {chunk_id}",
        )
        for rank, chunk_id in enumerate(order, 1)
    )
    candidates = tuple(candidate(chunk_id, rank) for rank, chunk_id in enumerate(order, 1))
    return request, SimpleNamespace(
        dense=candidates, bm25=candidates, fused=candidates, results=results
    )


def _batch_trace(
    experiment_id: str,
    *,
    tools: tuple[tuple[str, dict[str, object], bool], ...] = (),
    retrieval_order: tuple[str, ...] | None = None,
    rerank: bool = True,
    candidate: str | None = None,
) -> Trace:
    before = _trace_ids(experiment_id)
    with agent_invocation_span(request_origin="batch", eval_case_id="case-1") as root:
        for index, (name, arguments, succeeds) in enumerate(tools):
            with tool_execution_span(
                name=name, arguments=arguments, call_id=f"call-{index}"
            ) as tool:
                if name == "search_corpus" and retrieval_order is not None:
                    request, stages = _retrieval_data(retrieval_order, rerank=rerank)
                    with retrieval_span(request) as retriever:
                        record_retrieval_results(retriever, stages)
                record_tool_result(
                    tool,
                    {"ok": True, "data": {}}
                    if succeeds
                    else {"ok": False, "error": {"type": "expected_failure"}},
                )
        if candidate is not None:
            messages = (
                Message("m-1", "c-1", MessageRole.ASSISTANT, candidate, NOW),
            )
            record_batch_candidate_answer(root, messages)
    return _stored_trace(experiment_id, before)


def test_tool_trajectory_empty_and_immutable(trace_experiment: str) -> None:
    trace = _batch_trace(trace_experiment)

    assert trace_to_tool_calls(trace) == []
    call = EvaluationToolCall("get_draft", {"artifact_id": "draft-1"})
    with pytest.raises(TypeError):
        call.arguments["artifact_id"] = "changed"  # type: ignore[index]


def test_tool_trajectory_orders_exact_repeated_and_failed_calls(
    trace_experiment: str,
) -> None:
    trace = _batch_trace(
        trace_experiment,
        tools=(
            ("search_corpus", SEARCH_ARGUMENTS, True),
            ("get_draft", {"artifact_id": "draft-1"}, False),
            ("get_draft", {"artifact_id": "draft-2"}, True),
        ),
        retrieval_order=("chunk-2", "chunk-1"),
    )
    trace.data.spans.reverse()

    calls = trace_to_tool_calls(trace)

    assert [(call.tool, dict(call.arguments)) for call in calls] == [
        ("search_corpus", SEARCH_ARGUMENTS),
        ("get_draft", {"artifact_id": "draft-1"}),
        ("get_draft", {"artifact_id": "draft-2"}),
    ]


def test_malformed_tool_span_fails_explicitly(trace_experiment: str) -> None:
    trace = _batch_trace(
        trace_experiment,
        tools=(("search_corpus", {"query": "incomplete"}, True),),
    )

    with pytest.raises(TraceAdapterError, match="incomplete validated arguments"):
        trace_to_tool_calls(trace)


@pytest.mark.parametrize(
    ("order", "rerank"),
    [(("chunk-2", "chunk-1"), True), (("chunk-1", "chunk-2"), False)],
)
def test_retrieval_adapter_uses_stored_final_order_and_existing_scorer(
    trace_experiment: str, order: tuple[str, ...], rerank: bool
) -> None:
    trace = _batch_trace(
        trace_experiment,
        tools=(("search_corpus", {**SEARCH_ARGUMENTS, "rerank": rerank}, True),),
        retrieval_order=order,
        rerank=rerank,
    )
    references = {"case-1": RetrievalReference("case-1", frozenset({"chunk-1"}))}

    adapted = trace_to_retrieval_scorer_input(trace, references)

    assert adapted.predictions == order
    assert metrics_at_k(adapted.predictions, adapted.golden, 2) == metrics_at_k(
        order, frozenset({"chunk-1"}), 2
    )


def test_retrieval_adapter_handles_no_retrieval_and_rejects_missing_or_join(
    trace_experiment: str,
) -> None:
    references = {"case-1": RetrievalReference("case-1", frozenset({"chunk-1"}))}
    assert trace_to_retrieval_scorer_input(
        _batch_trace(trace_experiment), references
    ).predictions == ()
    missing = _batch_trace(
        trace_experiment, tools=(("search_corpus", SEARCH_ARGUMENTS, True),)
    )
    with pytest.raises(TraceAdapterError, match="missing its RETRIEVER"):
        trace_to_retrieval_scorer_input(missing, references)
    with pytest.raises(TraceAdapterError, match="no valid reference"):
        trace_to_retrieval_scorer_input(missing, {})


def test_generation_adapter_reconstructs_exact_existing_judge_inputs(
    trace_experiment: str,
) -> None:
    trace = _batch_trace(
        trace_experiment,
        tools=(("search_corpus", SEARCH_ARGUMENTS, True),),
        retrieval_order=("chunk-2", "chunk-1"),
        candidate="The final Aurora answer.",
    )
    golden_contexts = (GenerationContext("gold-1", "artifact-g", "gold context"),)
    references = {
        "case-1": GenerationReference("case-1", "Golden answer", golden_contexts)
    }

    adapted = trace_to_generation_judge_input(trace, references)

    expected_contexts = (
        GenerationContext("chunk-2", "artifact-2", "context for chunk-2"),
        GenerationContext("chunk-1", "artifact-1", "context for chunk-1"),
    )
    assert adapted.query == "Aurora launch"
    assert adapted.candidate_answer == "The final Aurora answer."
    assert adapted.retrieved_contexts == expected_contexts
    assert adapted.golden_answer == "Golden answer"
    assert adapted.golden_contexts == golden_contexts
    adapted_prompt = judge_prompt(
        GenerationMetric.FAITHFULNESS,
        query=adapted.query,
        candidate_answer=adapted.candidate_answer,
        golden_answer=adapted.golden_answer,
        retrieved_contexts=adapted.retrieved_contexts,
        golden_contexts=adapted.golden_contexts,
    )
    direct_prompt = judge_prompt(
        GenerationMetric.FAITHFULNESS,
        query="Aurora launch",
        candidate_answer="The final Aurora answer.",
        golden_answer="Golden answer",
        retrieved_contexts=expected_contexts,
        golden_contexts=golden_contexts,
    )
    assert adapted_prompt == direct_prompt


def test_generation_adapter_rejects_missing_batch_content_and_reference(
    trace_experiment: str,
) -> None:
    trace = _batch_trace(
        trace_experiment,
        tools=(("search_corpus", SEARCH_ARGUMENTS, True),),
        retrieval_order=("chunk-1",),
    )
    references = {
        "case-1": GenerationReference("case-1", "Golden", ())
    }
    with pytest.raises(TraceAdapterError, match="candidate answer"):
        trace_to_generation_judge_input(trace, references)
    with pytest.raises(TraceAdapterError, match="no valid reference"):
        trace_to_generation_judge_input(trace, {})
