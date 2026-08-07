"""Centralized, privacy-preserving MLflow tracing integration.

Traces retain operational metadata, model identity, token usage, latency, and
sanitized status. Raw prompts, conversation content, drafts, model output, provider
exceptions, credentials, and environment values are intentionally excluded.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Literal

import mlflow
from mlflow.entities import SpanStatusCode, SpanType

RequestOrigin = Literal["ui", "api", "batch"]
REQUEST_ORIGINS = frozenset({"ui", "api", "batch"})

# Stable stored-trace schema shared by instrumentation and pure evaluation adapters.
ATTR_REQUEST_ORIGIN = "request_origin"
ATTR_EVAL_CASE_ID = "eval_case_id"
ATTR_EVAL_RUN_NUMBER = "evaluation.run_number"
ATTR_EVAL_AGENT_TEMPERATURE = "evaluation.agent_temperature"
ATTR_CANDIDATE_ANSWER = "evaluation.candidate_answer"
ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_ARGUMENTS = "tool.arguments"
ATTR_RETRIEVAL_REQUEST = "retrieval.request"
ATTR_RETRIEVAL_FINAL_RESULTS = "retrieval.final_results"
ATTR_RETRIEVAL_CONTEXTS = "retrieval.contexts"
ATTR_RETRIEVAL_STAGE_RANKINGS = "retrieval.stage_rankings"

_initialization_lock = Lock()
_initialized = False
_invocation_origin: ContextVar[RequestOrigin | None] = ContextVar(
    "mlflow_invocation_origin", default=None
)


def initialize_mlflow_tracing() -> None:
    """Initialize local MLflow and framework instrumentation exactly once."""

    global _initialized
    with _initialization_lock:
        if _initialized:
            return
        tracking_uri = os.getenv(
            "EDITORIAL_MLFLOW_TRACKING_URI", "sqlite:///runtime_data/mlflow.db"
        ).strip()
        experiment_name = os.getenv(
            "EDITORIAL_MLFLOW_EXPERIMENT", "editorial-team"
        ).strip()
        tracking_uri = tracking_uri or "sqlite:///runtime_data/mlflow.db"
        if tracking_uri == "sqlite:///runtime_data/mlflow.db":
            Path("runtime_data").mkdir(parents=True, exist_ok=True, mode=0o700)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name or "editorial-team")
        mlflow.tracing.configure(span_processors=[_redact_span_content])
        mlflow.autolog(
            log_input_examples=False,
            log_model_signatures=False,
            log_models=False,
            log_datasets=False,
            log_traces=True,
            silent=True,
        )
        _initialized = True


def validate_request_origin(value: object) -> RequestOrigin:
    """Return one supported request origin without inferring it from identity."""

    if not isinstance(value, str) or value not in REQUEST_ORIGINS:
        raise ValueError("request_origin must be ui, api, or batch")
    return value  # type: ignore[return-value]


@contextmanager
def agent_invocation_span(
    *,
    request_origin: RequestOrigin,
    eval_case_id: str | None,
    eval_run_number: int | None = None,
    eval_agent_temperature: float | None = None,
):
    """Create the canonical root span when tracing was initialized by composition."""

    if not _initialized:
        yield None
        return
    attributes: dict[str, Any] = {
        ATTR_REQUEST_ORIGIN: request_origin,
        "eval_case_id_present": eval_case_id is not None,
    }
    if eval_case_id is not None:
        attributes[ATTR_EVAL_CASE_ID] = eval_case_id
    if request_origin == "batch" and eval_run_number is not None:
        attributes[ATTR_EVAL_RUN_NUMBER] = eval_run_number
    if request_origin == "batch" and eval_agent_temperature is not None:
        attributes[ATTR_EVAL_AGENT_TEMPERATURE] = eval_agent_temperature
    started = perf_counter()
    with mlflow.start_span(
        name="editorial_team.conversation_invocation",
        span_type=SpanType.AGENT,
        attributes=attributes,
    ) as span:
        origin_token = _invocation_origin.set(request_origin)
        tags = {ATTR_REQUEST_ORIGIN: request_origin}
        if eval_case_id is not None:
            tags[ATTR_EVAL_CASE_ID] = eval_case_id
        mlflow.update_current_trace(tags=tags)
        try:
            yield span
        finally:
            span.set_attribute("latency_ms", (perf_counter() - started) * 1000.0)
            _invocation_origin.reset(origin_token)


@contextmanager
def gemini_llm_span(*, model: str):
    """Create one metadata-only span for the direct Google SDK model boundary."""

    if not _initialized:
        yield None
        return
    started = perf_counter()
    with mlflow.start_span(
        name="gemini.interactions.create",
        span_type=SpanType.CHAT_MODEL,
        attributes={
            "mlflow.llm.model": model,
            "gen_ai.request.model": model,
        },
    ) as span:
        try:
            yield span
        finally:
            span.set_attribute("latency_ms", (perf_counter() - started) * 1000.0)


@contextmanager
def langchain_llm_span(*, model: object):
    """Trace the installed LangChain-core chat boundary when autolog cannot patch it."""

    configured = getattr(model, "model", None)
    model_name = configured if isinstance(configured, str) else type(model).__name__
    with gemini_llm_span(model=model_name) as span:
        yield span


def record_langchain_usage(span: Any, response: Any) -> None:
    """Normalize real LangChain response usage without retaining message content."""

    if span is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        record_gemini_usage(span, response)
        return
    normalized = SimpleUsageMetadata(
        prompt_token_count=usage.get("input_tokens"),
        response_token_count=usage.get("output_tokens"),
        total_token_count=usage.get("total_tokens"),
    )
    record_gemini_usage(span, SimpleNamespace(usage_metadata=normalized))


class SimpleUsageMetadata:
    """Private attribute carrier for provider-neutral usage normalization."""

    def __init__(
        self,
        *,
        prompt_token_count: object,
        response_token_count: object,
        total_token_count: object,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.response_token_count = response_token_count
        self.total_token_count = total_token_count


def record_gemini_usage(span: Any, interaction: Any) -> None:
    """Record real provider counts and mark unavailable counts without estimating."""

    if span is None:
        return
    usage = getattr(interaction, "usage_metadata", None)
    input_tokens = _optional_count(usage, "prompt_token_count")
    output_tokens = _optional_count(usage, "response_token_count")
    if output_tokens is None:
        output_tokens = _optional_count(usage, "candidates_token_count")
    total_tokens = _optional_count(usage, "total_token_count")
    available = {
        "input": input_tokens is not None,
        "output": output_tokens is not None,
        "total": total_tokens is not None,
    }
    span.set_attributes(
        {
            "token_count.input_available": available["input"],
            "token_count.output_available": available["output"],
            "token_count.total_available": available["total"],
        }
    )
    token_usage = {
        key: value
        for key, value in {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }.items()
        if value is not None
    }
    if token_usage:
        span.set_attribute("mlflow.chat.tokenUsage", token_usage)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)


def mark_llm_failure(span: Any) -> None:
    """Mark a model span failed using a fixed description-free status."""

    if span is not None:
        span.set_status(SpanStatusCode.ERROR)
        span.set_attribute("error.type", "provider_model_failure")


@contextmanager
def tool_execution_span(
    *, name: str, arguments: object, call_id: object
):
    """Create a TOOL span with an origin-aware allowlist of model-visible arguments."""

    if not _trace_child_spans():
        yield None
        return
    started = perf_counter()
    attributes: dict[str, Any] = {
        ATTR_TOOL_NAME: name,
        ATTR_TOOL_ARGUMENTS: _safe_tool_arguments(name, arguments),
    }
    if name == "search_corpus":
        attributes["tool.query_retained"] = _invocation_origin.get() == "batch"
    if isinstance(call_id, str) and call_id.strip():
        attributes["tool.call_id"] = call_id
    with mlflow.start_span(
        name=name,
        span_type=SpanType.TOOL,
        attributes=attributes,
    ) as span:
        try:
            yield span
        finally:
            span.set_attribute("latency_ms", (perf_counter() - started) * 1000.0)


def record_batch_candidate_answer(span: Any, messages: object) -> None:
    """Retain only the final delivered batch response required by generation judges."""

    if span is None or _invocation_origin.get() != "batch":
        return
    if not isinstance(messages, tuple) or not messages:
        return
    contents = [getattr(message, "content", None) for message in messages]
    if all(isinstance(content, str) for content in contents):
        span.set_attribute(ATTR_CANDIDATE_ANSWER, "\n\n".join(contents))


def record_tool_result(span: Any, output: object) -> None:
    """Record bounded status for structured tool results, including logical failures."""

    if span is None:
        return
    if isinstance(output, dict) and output.get("ok") is True:
        span.set_attributes(
            {
                "tool.success": True,
                "tool.failure_kind": "none",
                **_bounded_tool_result(output),
            }
        )
        return
    error_type = "tool_failure"
    if isinstance(output, dict):
        error = output.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            error_type = error["type"]
    span.set_status(SpanStatusCode.ERROR)
    span.set_attributes(
        {
            "tool.success": False,
            "tool.failure_kind": "logical",
            "tool.error_type": error_type,
        }
    )


def record_tool_exception(span: Any) -> None:
    """Record a sanitized actual tool exception."""

    if span is not None:
        span.set_status(SpanStatusCode.ERROR)
        span.set_attributes(
            {
                "tool.success": False,
                "tool.failure_kind": "exception",
                "tool.error_type": "tool_execution_exception",
            }
        )


@contextmanager
def retrieval_span(request: Any):
    """Create one RETRIEVER span nested beneath the active search tool."""

    if not _trace_child_spans():
        yield None
        return
    started = perf_counter()
    with mlflow.start_span(
        name="corpus_retrieval",
        span_type=SpanType.RETRIEVER,
        attributes={ATTR_RETRIEVAL_REQUEST: _retrieval_request(request)},
    ) as span:
        try:
            yield span
        finally:
            span.set_attribute("latency_ms", (perf_counter() - started) * 1000.0)


def record_retrieval_results(span: Any, stages: Any) -> None:
    """Retain ordered rankings and origin-appropriate bounded contexts."""

    if span is None:
        return
    final_results = [
        {
            "chunk_id": item.chunk_id,
            "artifact_id": item.artifact_id,
            "rank": item.rank,
            "dense_score": item.dense_score,
            "bm25_score": item.bm25_score,
            "rrf_score": item.rrf_score,
            "rerank_score": item.rerank_score,
        }
        for item in stages.results
    ]
    stage_rankings = {
        "dense": [item.chunk.chunk.chunk_id for item in stages.dense],
        "bm25": [item.chunk.chunk.chunk_id for item in stages.bm25],
        "rrf": [item.chunk.chunk.chunk_id for item in stages.fused],
        "final": [item.chunk_id for item in stages.results],
    }
    retain_content = _invocation_origin.get() == "batch"
    attributes: dict[str, Any] = {
        "retrieval.success": True,
        ATTR_RETRIEVAL_FINAL_RESULTS: final_results,
        ATTR_RETRIEVAL_STAGE_RANKINGS: stage_rankings,
        "retrieval.contexts_retained": retain_content,
    }
    if retain_content:
        attributes[ATTR_RETRIEVAL_CONTEXTS] = [
            {
                "chunk_id": item.chunk_id,
                "artifact_id": item.artifact_id,
                "content": item.excerpt,
            }
            for item in stages.results
        ]
    span.set_attributes(attributes)


def record_retrieval_failure(span: Any) -> None:
    """Mark retrieval failure without provider, database, or content details."""

    if span is not None:
        span.set_status(SpanStatusCode.ERROR)
        span.set_attributes(
            {"retrieval.success": False, "retrieval.error_type": "retrieval_failure"}
        )


def _optional_count(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _trace_child_spans() -> bool:
    return _initialized and _invocation_origin.get() is not None


def _safe_tool_arguments(name: str, arguments: object) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    if name == "search_corpus":
        allowed = {
            "created_from",
            "created_to",
            "prefer_recent",
            "top_k",
            "rerank",
        }
        output = {key: arguments[key] for key in allowed if key in arguments}
        retain_query = _invocation_origin.get() == "batch"
        if retain_query and isinstance(arguments.get("query"), str):
            output["query"] = arguments["query"]
        return output
    if name == "get_draft":
        artifact_id = arguments.get("artifact_id")
        return {"artifact_id": artifact_id} if isinstance(artifact_id, str) else {}
    return {}


def _bounded_tool_result(output: dict[str, Any]) -> dict[str, Any]:
    data = output.get("data")
    if not isinstance(data, dict):
        return {}
    results = data.get("results")
    if isinstance(results, list):
        return {"tool.result_count": len(results)}
    artifact_id = data.get("artifact_id")
    if isinstance(artifact_id, str):
        return {"tool.result_artifact_id": artifact_id}
    return {}


def _retrieval_request(request: Any) -> dict[str, Any]:
    retain_query = _invocation_origin.get() == "batch"
    value: dict[str, Any] = {
        "query_retained": retain_query,
        "created_from": _timestamp(getattr(request, "created_from", None)),
        "created_to": _timestamp(getattr(request, "created_to", None)),
        "prefer_recent": getattr(request, "prefer_recent", False),
        "top_k": getattr(request, "top_k", None),
        "rerank": getattr(request, "rerank", False),
    }
    query = getattr(request, "query", None)
    if retain_query and isinstance(query, str):
        value["query"] = query
    return value


def _timestamp(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _redact_span_content(span: Any) -> None:
    """Remove framework-captured product content immediately before trace export."""

    if span.inputs is not None:
        span.set_inputs({"content_retained": False})
    if span.outputs is not None:
        span.set_outputs({"content_retained": False})
