from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
from mlflow.entities import SpanStatusCode, SpanType

from editorial_team.conversation import ConversationService, ConversationServiceError
from editorial_team.domain.conversation import Message, MessageRole
from editorial_team.gemini import GeminiModelClient
from editorial_team.mlflow_tracing import (
    agent_invocation_span,
    initialize_mlflow_tracing,
    record_retrieval_results,
    record_tool_result,
    retrieval_span,
    tool_execution_span,
)
from editorial_team.models import ModelRequest

NOW = datetime(2026, 8, 7, tzinfo=UTC)
SECRET = "TRACE-SECRET-CANARY"


class FakeInteractions:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def create(self, **kwargs: object) -> object:
        del kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            id="interaction-1",
            output_text="safe response",
            steps=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                response_token_count=7,
                total_token_count=18,
            ),
        )


class ModelGraph:
    def __init__(self, model: GeminiModelClient) -> None:
        self.model = model

    def invoke(self, state: object, config: object) -> dict[str, object]:
        del state, config
        self.model.respond(ModelRequest(f"private prompt {SECRET}"))
        return {
            "assistant_messages": (
                Message(
                    "message-1",
                    "conversation-1",
                    MessageRole.ASSISTANT,
                    "safe final response",
                    NOW,
                ),
            )
        }


@pytest.fixture
def trace_experiment(tmp_path: Path, request: pytest.FixtureRequest) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_name = f"stage-1-{request.node.name}"
    mlflow.set_experiment(experiment_name)
    initialize_mlflow_tracing()
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def traces(experiment_id: str) -> list[object]:
    return mlflow.search_traces(
        locations=[experiment_id],
        return_type="list",
        include_spans=True,
        flush=True,
    )


def test_conversation_trace_has_one_root_child_llm_metadata_and_redaction(
    trace_experiment: str,
) -> None:
    model = GeminiModelClient(
        model="gemini-test-model",
        sdk_client=SimpleNamespace(interactions=FakeInteractions()),
    )
    service = ConversationService(graph_runner=ModelGraph(model))
    before = {item.info.trace_id for item in traces(trace_experiment)}

    messages = service.process_message(
        "conversation-1",
        f"private input {SECRET}",
        request_origin="batch",
        eval_case_id="case-1",
        eval_run_number=2,
        eval_agent_temperature=0.2,
    )

    assert messages[0].content == "safe final response"
    captured = [
        item for item in traces(trace_experiment) if item.info.trace_id not in before
    ]
    assert len(captured) == 1
    trace = captured[0]
    roots = [span for span in trace.data.spans if span.parent_id is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.name == "editorial_team.conversation_invocation"
    assert root.span_type == SpanType.AGENT
    assert root.attributes["request_origin"] == "batch"
    assert root.attributes["eval_case_id"] == "case-1"
    assert root.attributes["evaluation.run_number"] == 2
    assert root.attributes["evaluation.agent_temperature"] == 0.2
    assert root.attributes["evaluation.candidate_answer"] == "safe final response"
    assert root.attributes["latency_ms"] >= 0
    assert trace.info.tags["request_origin"] == "batch"
    assert trace.info.tags["eval_case_id"] == "case-1"

    llm_spans = [span for span in trace.data.spans if span.span_type == SpanType.CHAT_MODEL]
    assert len(llm_spans) == 1
    llm = llm_spans[0]
    assert llm.parent_id == root.span_id
    assert llm.attributes["mlflow.llm.model"] == "gemini-test-model"
    assert llm.attributes["mlflow.chat.tokenUsage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert llm.attributes["gen_ai.usage.input_tokens"] == 11
    assert llm.attributes["gen_ai.usage.output_tokens"] == 7
    assert llm.attributes["latency_ms"] >= 0
    assert llm.end_time_ns >= llm.start_time_ns
    assert SECRET not in trace.to_json()


def test_eval_case_is_optional_and_default_origin_is_compatible(
    trace_experiment: str,
) -> None:
    message = Message(
        "message-1", "conversation-1", MessageRole.ASSISTANT, "Reply", NOW
    )
    graph = SimpleNamespace(
        invoke=lambda state, config: {"assistant_messages": (message,)}
    )
    before = {item.info.trace_id for item in traces(trace_experiment)}

    ConversationService(graph_runner=graph).process_message("conversation-1", "Hello")

    captured = [
        item for item in traces(trace_experiment) if item.info.trace_id not in before
    ]
    assert len(captured) == 1
    trace = captured[0]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.attributes["request_origin"] == "api"
    assert root.attributes["eval_case_id_present"] is False
    assert "eval_case_id" not in root.attributes
    assert "evaluation.candidate_answer" not in root.attributes


def test_failed_gemini_call_is_sanitized_and_marked_error(trace_experiment: str) -> None:
    model = GeminiModelClient(
        model="gemini-test-model",
        sdk_client=SimpleNamespace(
            interactions=FakeInteractions(error=RuntimeError(f"provider details {SECRET}"))
        ),
    )
    service = ConversationService(graph_runner=ModelGraph(model))
    before = {item.info.trace_id for item in traces(trace_experiment)}

    with pytest.raises(ConversationServiceError, match="Conversation graph failed"):
        service.process_message("conversation-1", "Hello", request_origin="ui")

    captured = [
        item for item in traces(trace_experiment) if item.info.trace_id not in before
    ]
    assert len(captured) == 1
    trace = captured[0]
    llm = next(span for span in trace.data.spans if span.span_type == SpanType.CHAT_MODEL)
    assert llm.status.status_code is SpanStatusCode.ERROR
    assert llm.attributes["error.type"] == "provider_model_failure"
    assert SECRET not in trace.to_json()


def retrieval_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    request = SimpleNamespace(
        query="Aurora launch",
        created_from=datetime(2026, 1, 1, tzinfo=UTC),
        created_to=None,
        prefer_recent=True,
        top_k=2,
        rerank=True,
    )

    def candidate(chunk_id: str, rank: int) -> SimpleNamespace:
        stored = SimpleNamespace(chunk=SimpleNamespace(chunk_id=chunk_id))
        return SimpleNamespace(chunk=stored, rank=rank)

    final = (
        SimpleNamespace(
            chunk_id="chunk-2",
            artifact_id="artifact-2",
            rank=1,
            dense_score=0.8,
            bm25_score=2.0,
            rrf_score=0.03,
            rerank_score=0.9,
            excerpt="Bounded Aurora context two",
        ),
        SimpleNamespace(
            chunk_id="chunk-1",
            artifact_id="artifact-1",
            rank=2,
            dense_score=0.7,
            bm25_score=1.0,
            rrf_score=0.02,
            rerank_score=0.8,
            excerpt="Bounded Aurora context one",
        ),
    )
    stages = SimpleNamespace(
        dense=(candidate("chunk-1", 1), candidate("chunk-2", 2)),
        bm25=(candidate("chunk-2", 1), candidate("chunk-1", 2)),
        fused=(candidate("chunk-2", 1), candidate("chunk-1", 2)),
        results=final,
    )
    return request, stages


def test_batch_tool_and_retrieval_spans_preserve_order_arguments_and_contexts(
    trace_experiment: str,
) -> None:
    request, stages = retrieval_fixture()
    before = {item.info.trace_id for item in traces(trace_experiment)}
    search_arguments = {
        "query": "Aurora launch",
        "created_from": "2026-01-01T00:00:00+00:00",
        "created_to": None,
        "prefer_recent": True,
        "top_k": 2,
        "rerank": True,
    }

    with agent_invocation_span(request_origin="batch", eval_case_id="case-tools"):
        with tool_execution_span(
            name="search_corpus", arguments=search_arguments, call_id="call-1"
        ) as search:
            with retrieval_span(request) as retrieval:
                record_retrieval_results(retrieval, stages)
            record_tool_result(search, {"ok": True, "data": {"results": [{}, {}]}})
        with tool_execution_span(
            name="get_draft",
            arguments={"artifact_id": "artifact-2"},
            call_id="call-2",
        ) as get:
            record_tool_result(
                get,
                {
                    "ok": True,
                    "data": {"artifact_id": "artifact-2", "content": SECRET},
                },
            )

    captured = [
        item for item in traces(trace_experiment) if item.info.trace_id not in before
    ]
    assert len(captured) == 1
    trace = captured[0]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    tools = sorted(
        (span for span in trace.data.spans if span.span_type == SpanType.TOOL),
        key=lambda span: span.start_time_ns,
    )
    assert [span.name for span in tools] == ["search_corpus", "get_draft"]
    assert all(span.parent_id == root.span_id for span in tools)
    assert tools[0].attributes["tool.call_id"] == "call-1"
    assert tools[0].attributes["tool.arguments"] == search_arguments
    assert tools[0].attributes["tool.query_retained"] is True
    assert "conversation_id" not in tools[0].attributes["tool.arguments"]
    assert tools[1].attributes["tool.arguments"] == {"artifact_id": "artifact-2"}
    assert tools[1].attributes["tool.result_artifact_id"] == "artifact-2"

    retrievers = [
        span for span in trace.data.spans if span.span_type == SpanType.RETRIEVER
    ]
    assert len(retrievers) == 1
    retriever = retrievers[0]
    assert retriever.parent_id == tools[0].span_id
    assert retriever.attributes["retrieval.request"]["rerank"] is True
    assert retriever.attributes["retrieval.request"]["query"] == "Aurora launch"
    assert retriever.attributes["retrieval.stage_rankings"] == {
        "dense": ["chunk-1", "chunk-2"],
        "bm25": ["chunk-2", "chunk-1"],
        "rrf": ["chunk-2", "chunk-1"],
        "final": ["chunk-2", "chunk-1"],
    }
    assert [
        (item["chunk_id"], item["artifact_id"], item["rank"])
        for item in retriever.attributes["retrieval.final_results"]
    ] == [("chunk-2", "artifact-2", 1), ("chunk-1", "artifact-1", 2)]
    assert retriever.attributes["retrieval.contexts"] == [
        {
            "chunk_id": "chunk-2",
            "artifact_id": "artifact-2",
            "content": "Bounded Aurora context two",
        },
        {
            "chunk_id": "chunk-1",
            "artifact_id": "artifact-1",
            "content": "Bounded Aurora context one",
        },
    ]
    assert SECRET not in trace.to_json()


def test_ui_policy_redacts_query_and_retrieved_context(trace_experiment: str) -> None:
    request, stages = retrieval_fixture()
    request.query = SECRET
    stages.results[0].excerpt = SECRET
    before = {item.info.trace_id for item in traces(trace_experiment)}

    with agent_invocation_span(request_origin="ui", eval_case_id=None):
        with tool_execution_span(
            name="search_corpus", arguments={"query": SECRET, "top_k": 2}, call_id="call-ui"
        ) as search:
            with retrieval_span(request) as retrieval:
                record_retrieval_results(retrieval, stages)
            record_tool_result(search, {"ok": True, "data": {"results": [{}, {}]}})

    trace = next(
        item
        for item in traces(trace_experiment)
        if item.info.trace_id not in before
    )
    tool = next(span for span in trace.data.spans if span.span_type == SpanType.TOOL)
    retriever = next(
        span for span in trace.data.spans if span.span_type == SpanType.RETRIEVER
    )
    assert tool.attributes["tool.arguments"] == {"top_k": 2}
    assert tool.attributes["tool.query_retained"] is False
    assert retriever.attributes["retrieval.request"]["query_retained"] is False
    assert retriever.attributes["retrieval.contexts_retained"] is False
    assert "retrieval.contexts" not in retriever.attributes
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert "evaluation.candidate_answer" not in root.attributes
    assert SECRET not in trace.to_json()


def test_logical_tool_failure_has_error_status_and_stable_category(
    trace_experiment: str,
) -> None:
    before = {item.info.trace_id for item in traces(trace_experiment)}
    with agent_invocation_span(request_origin="batch", eval_case_id="case-failure"):
        with tool_execution_span(
            name="get_draft", arguments={"artifact_id": "missing"}, call_id="call-fail"
        ) as span:
            record_tool_result(
                span,
                {
                    "ok": False,
                    "error": {"type": "artifact_not_found", "message": SECRET},
                },
            )

    trace = next(
        item
        for item in traces(trace_experiment)
        if item.info.trace_id not in before
    )
    tool = next(span for span in trace.data.spans if span.span_type == SpanType.TOOL)
    assert tool.status.status_code is SpanStatusCode.ERROR
    assert tool.attributes["tool.success"] is False
    assert tool.attributes["tool.failure_kind"] == "logical"
    assert tool.attributes["tool.error_type"] == "artifact_not_found"
    assert SECRET not in trace.to_json()
