"""Conversation-scoped real LangChain tools for editorial artifact retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from editorial_team.artifacts.retrieval import HybridRetriever
from editorial_team.artifacts.retrieval_types import SearchResult
from editorial_team.contracts.common import parse_utc_timestamp, timestamp_to_json
from editorial_team.contracts.identity import validate_identifier


class SearchCorpusInput(BaseModel):
    """Strict model-visible corpus-search arguments."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Words describing the previous writing")
    created_from: str | None = Field(
        default=None,
        description="Inclusive UTC time when Editorial Team produced the draft",
    )
    created_to: str | None = Field(
        default=None,
        description="Inclusive UTC time when Editorial Team produced the draft",
    )
    prefer_recent: bool = Field(
        default=False,
        description="Use creation time only to break ties after relevance",
    )
    top_k: int = Field(default=5, ge=1, le=10)
    rerank: bool = Field(default=False)


class GetDraftInput(BaseModel):
    """Strict model-visible complete-draft arguments."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)


def build_editorial_retrieval_tools(
    *, retriever: HybridRetriever, conversation_id: str
) -> tuple[StructuredTool, StructuredTool]:
    """Build exactly two real LangChain tools closed over a validated scope."""

    conversation_id = validate_identifier(conversation_id, "conversation_id")

    def search_corpus(
        query: str,
        created_from: str | None = None,
        created_to: str | None = None,
        prefer_recent: bool = False,
        top_k: int = 5,
        rerank: bool = False,
    ) -> dict[str, Any]:
        try:
            lower = (
                None
                if created_from is None
                else parse_utc_timestamp(created_from, "created_from")
            )
            upper = (
                None if created_to is None else parse_utc_timestamp(created_to, "created_to")
            )
            results = retriever.search(
                query=query,
                conversation_id=conversation_id,
                created_from=lower,
                created_to=upper,
                prefer_recent=prefer_recent,
                top_k=top_k,
                rerank=rerank,
            )
            return {
                "ok": True,
                "data": {
                    "query": query.strip(),
                    "created_from": None if lower is None else timestamp_to_json(lower),
                    "created_to": None if upper is None else timestamp_to_json(upper),
                    "prefer_recent": prefer_recent,
                    "rerank": rerank,
                    "results": [_search_result(item) for item in results],
                },
            }
        except Exception:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_search_request",
                    "message": "The corpus search request is invalid",
                },
            }

    def get_draft(artifact_id: str) -> dict[str, Any]:
        try:
            draft = retriever.get_draft(
                artifact_id=artifact_id,
                conversation_id=conversation_id,
            )
        except Exception:
            return {
                "ok": False,
                "error": {
                    "type": "artifact_not_found",
                    "message": "The draft artifact was not found",
                },
            }
        if draft is None:
            return {
                "ok": False,
                "error": {
                    "type": "artifact_not_found",
                    "message": "The draft artifact was not found",
                },
            }
        return {
            "ok": True,
            "data": {
                "artifact_id": draft.artifact_id,
                "task_id": draft.task_id,
                "producer": draft.producer.value,
                "created_at": timestamp_to_json(draft.created_at),
                "user_request": draft.user_request,
                "content": draft.content,
            },
        }

    search_tool = StructuredTool.from_function(
        func=search_corpus,
        name="search_corpus",
        description=(
            "Search excerpts from previous Writer and Editor outputs in the current "
            "conversation. Use this when the user refers to previous writing. Refine and "
            "repeat searches when useful. Time bounds mean when Editorial Team produced the "
            "draft. Search does not load complete drafts; after choosing an artifact, call "
            "get_draft explicitly."
        ),
        args_schema=SearchCorpusInput,
    )
    draft_tool = StructuredTool.from_function(
        func=get_draft,
        name="get_draft",
        description=(
            "Load one complete immutable draft by artifact ID inside the current conversation. "
            "Call only after search_corpus identifies and you select a candidate."
        ),
        args_schema=GetDraftInput,
    )
    return search_tool, draft_tool


def _search_result(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "artifact_id": result.artifact_id,
        "task_id": result.task_id,
        "excerpt": result.excerpt,
        "created_at": timestamp_to_json(result.created_at),
        "producer": result.producer.value,
        "dense_rank": result.dense_rank,
        "dense_score": result.dense_score,
        "bm25_rank": result.bm25_rank,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "rerank_score": result.rerank_score,
        "chunk_ordinal": result.chunk_ordinal,
    }
