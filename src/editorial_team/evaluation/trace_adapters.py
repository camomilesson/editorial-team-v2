"""Pure adapters from stored MLflow traces to existing evaluation contracts.

Observed values come only from the stored trace. Golden values come only from a
bounded reference mapping joined by the trace's ``eval_case_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mlflow.entities import SpanType

from editorial_team.evaluation.generation_models import GenerationContext
from editorial_team.mlflow_tracing import (
    ATTR_CANDIDATE_ANSWER,
    ATTR_EVAL_CASE_ID,
    ATTR_REQUEST_ORIGIN,
    ATTR_RETRIEVAL_CONTEXTS,
    ATTR_RETRIEVAL_FINAL_RESULTS,
    ATTR_RETRIEVAL_REQUEST,
    ATTR_TOOL_ARGUMENTS,
    ATTR_TOOL_NAME,
)

_TOOL_ARGUMENT_KEYS = {
    "search_corpus": frozenset(
        {"query", "created_from", "created_to", "prefer_recent", "top_k", "rerank"}
    ),
    "get_draft": frozenset({"artifact_id"}),
}


class TraceAdapterError(ValueError):
    """A stored trace cannot satisfy an existing scorer contract."""


@dataclass(frozen=True)
class EvaluationToolCall:
    """Minimal immutable tool trajectory item required by HW3 metrics."""

    tool: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tool not in _TOOL_ARGUMENT_KEYS:
            raise ValueError("tool must be an actual registered editorial tool")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping")
        copied = deepcopy(dict(self.arguments))
        object.__setattr__(self, "arguments", MappingProxyType(copied))


@dataclass(frozen=True)
class RetrievalReference:
    case_id: str
    golden_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalScorerInput:
    eval_case_id: str
    predictions: tuple[str, ...]
    golden: frozenset[str]


@dataclass(frozen=True)
class GenerationReference:
    case_id: str
    golden_answer: str
    golden_contexts: tuple[GenerationContext, ...]


@dataclass(frozen=True)
class GenerationJudgeInput:
    eval_case_id: str
    query: str
    candidate_answer: str
    golden_answer: str
    retrieved_contexts: tuple[GenerationContext, ...]
    golden_contexts: tuple[GenerationContext, ...]


def trace_to_tool_calls(trace: Any) -> list[EvaluationToolCall]:
    """Return TOOL spans in chronological order, ignoring all results and other spans."""

    tool_spans = sorted(
        (span for span in _spans(trace) if span.span_type == SpanType.TOOL),
        key=lambda span: (span.start_time_ns, span.span_id),
    )
    output: list[EvaluationToolCall] = []
    for span in tool_spans:
        name = span.attributes.get(ATTR_TOOL_NAME)
        arguments = span.attributes.get(ATTR_TOOL_ARGUMENTS)
        if not isinstance(name, str) or name not in _TOOL_ARGUMENT_KEYS:
            raise TraceAdapterError("TOOL span has an invalid registered tool name")
        if span.name != name:
            raise TraceAdapterError("TOOL span name does not match its registered tool name")
        if not isinstance(arguments, dict):
            raise TraceAdapterError("TOOL span is missing validated arguments")
        required = _TOOL_ARGUMENT_KEYS[name]
        if set(arguments) != required:
            raise TraceAdapterError(f"{name} TOOL span has incomplete validated arguments")
        output.append(EvaluationToolCall(name, arguments))
    return output


def trace_to_retrieval_scorer_input(
    trace: Any,
    references: Mapping[str, RetrievalReference],
) -> RetrievalScorerInput:
    """Join final runtime chunk order to golden chunk IDs without rerunning retrieval."""

    case_id = _eval_case_id(trace)
    reference = _reference(references, case_id, RetrievalReference)
    retrievers = _retriever_spans(trace)
    if not retrievers:
        if any(call.tool == "search_corpus" for call in trace_to_tool_calls(trace)):
            raise TraceAdapterError("search tool call is missing its RETRIEVER span")
        predictions: tuple[str, ...] = ()
    else:
        final_results = retrievers[-1].attributes.get(ATTR_RETRIEVAL_FINAL_RESULTS)
        if not isinstance(final_results, list):
            raise TraceAdapterError("RETRIEVER span is missing final results")
        predictions = tuple(_result_identifier(item, "chunk_id") for item in final_results)
        if len(set(predictions)) != len(predictions):
            raise TraceAdapterError("RETRIEVER final results contain duplicate chunk IDs")
    return RetrievalScorerInput(case_id, predictions, reference.golden_chunk_ids)


def trace_to_generation_judge_input(
    trace: Any,
    references: Mapping[str, GenerationReference],
) -> GenerationJudgeInput:
    """Build the existing generation-judge inputs from a stored batch trace and references."""

    case_id = _eval_case_id(trace)
    reference = _reference(references, case_id, GenerationReference)
    root = _root_span(trace)
    if root.attributes.get(ATTR_REQUEST_ORIGIN) != "batch":
        raise TraceAdapterError("generation judge input requires a batch trace")
    candidate = root.attributes.get(ATTR_CANDIDATE_ANSWER)
    if not isinstance(candidate, str) or not candidate.strip():
        raise TraceAdapterError("batch trace is missing the final candidate answer")
    retrievers = _retriever_spans(trace)
    if not retrievers:
        raise TraceAdapterError("generation trace is missing a RETRIEVER span")
    retriever = retrievers[-1]
    request = retriever.attributes.get(ATTR_RETRIEVAL_REQUEST)
    contexts = retriever.attributes.get(ATTR_RETRIEVAL_CONTEXTS)
    if not isinstance(request, dict) or not isinstance(request.get("query"), str):
        raise TraceAdapterError("batch RETRIEVER span is missing its query")
    if not isinstance(contexts, list):
        raise TraceAdapterError("batch RETRIEVER span is missing retained contexts")
    retrieved = tuple(_generation_context(value) for value in contexts)
    return GenerationJudgeInput(
        eval_case_id=case_id,
        query=request["query"],
        candidate_answer=candidate,
        golden_answer=reference.golden_answer,
        retrieved_contexts=retrieved,
        golden_contexts=reference.golden_contexts,
    )


def _spans(trace: Any) -> list[Any]:
    data = getattr(trace, "data", None)
    spans = getattr(data, "spans", None)
    if not isinstance(spans, list):
        raise TraceAdapterError("trace does not contain a stored span list")
    return spans


def _root_span(trace: Any) -> Any:
    roots = [span for span in _spans(trace) if span.parent_id is None]
    if len(roots) != 1 or roots[0].span_type != SpanType.AGENT:
        raise TraceAdapterError("trace must contain exactly one AGENT root span")
    return roots[0]


def _eval_case_id(trace: Any) -> str:
    value = _root_span(trace).attributes.get(ATTR_EVAL_CASE_ID)
    if not isinstance(value, str) or not value.strip():
        raise TraceAdapterError("trace is missing eval_case_id")
    return value


def _retriever_spans(trace: Any) -> list[Any]:
    return sorted(
        (span for span in _spans(trace) if span.span_type == SpanType.RETRIEVER),
        key=lambda span: (span.start_time_ns, span.span_id),
    )


def _reference(references: Mapping[str, Any], case_id: str, kind: type[Any]) -> Any:
    reference = references.get(case_id)
    if not isinstance(reference, kind) or reference.case_id != case_id:
        raise TraceAdapterError(f"no valid reference fixture exists for {case_id}")
    return reference


def _result_identifier(value: object, key: str) -> str:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get(key), str)
        or not value[key].strip()
    ):
        raise TraceAdapterError("RETRIEVER result is malformed")
    return value[key]


def _generation_context(value: object) -> GenerationContext:
    if not isinstance(value, dict) or set(value) != {"chunk_id", "artifact_id", "content"}:
        raise TraceAdapterError("retained generation context is malformed")
    fields = (value["chunk_id"], value["artifact_id"], value["content"])
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise TraceAdapterError("retained generation context is malformed")
    return GenerationContext(
        chunk_id=value["chunk_id"],
        artifact_id=value["artifact_id"],
        content=value["content"],
    )
