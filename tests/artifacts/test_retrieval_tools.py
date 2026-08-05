"""LangChain tool construction, scope, and structured-output tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    HybridRetriever,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
    build_editorial_retrieval_tools,
    content_sha256,
)

NOW = datetime(2026, 8, 5, 12, tzinfo=UTC)


class Embeddings:
    model_id = "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1) / 10] for index, _text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


@dataclass
class Reranker:
    fail: bool = False

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        if self.fail:
            raise RuntimeError("private model failure")
        return [float(len(passage)) for passage in passages]


def artifact(task_id: str, conversation_id: str, content: str) -> EditorialArtifact:
    return EditorialArtifact(
        artifact_id=artifact_id_for(task_id, ArtifactProducer.WRITER),
        task_id=task_id,
        producer=ArtifactProducer.WRITER,
        created_at=NOW,
        conversation_id=conversation_id,
        user_request="Original request",
        content=content,
        content_sha256=content_sha256(content),
    )


@pytest.fixture
def tools(tmp_path: Path) -> tuple[object, object, EditorialArtifact, EditorialArtifact]:
    store = SQLiteArtifactStore(tmp_path / "artifacts.db", chunker=ParagraphChunker())
    store.initialize()
    visible = artifact("visible", "conversation-1", "Visible complete draft")
    hidden = artifact("hidden", "conversation-2", "Secret complete draft")
    store.save_run((visible,))
    store.save_run((hidden,))
    retriever = HybridRetriever(
        store=store,
        embeddings=Embeddings(),
        reranker=Reranker(),
        dense_depth=10,
        bm25_depth=10,
        fused_depth=10,
        rerank_depth=10,
    )
    search, get = build_editorial_retrieval_tools(
        retriever=retriever, conversation_id="conversation-1"
    )
    yield search, get, visible, hidden
    store.close()


def test_tools_have_exact_names_strict_visible_schemas_and_hidden_scope(tools: tuple) -> None:
    search, get, _visible, _hidden = tools
    assert [search.name, get.name] == ["search_corpus", "get_draft"]
    search_schema = search.args_schema.model_json_schema()
    get_schema = get.args_schema.model_json_schema()
    assert "conversation_id" not in search_schema["properties"]
    assert "conversation_id" not in get_schema["properties"]
    assert search_schema["additionalProperties"] is False
    assert get_schema["additionalProperties"] is False
    assert search_schema["properties"]["rerank"]["default"] is False
    with pytest.raises(ValidationError):
        search.invoke({"query": "draft", "conversation_id": "conversation-2"})


def test_search_is_repeatable_date_bounded_serializable_and_excerpt_only(tools: tuple) -> None:
    search, _get, visible, _hidden = tools
    arguments = {
        "query": "visible",
        "created_from": "2026-08-05T00:00:00Z",
        "created_to": "2026-08-05T23:59:59Z",
        "rerank": False,
    }
    first = search.invoke(arguments)
    second = search.invoke(arguments | {"query": "complete draft"})
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["data"]["rerank"] is False
    assert first["data"]["results"][0]["artifact_id"] == visible.artifact_id
    assert "content" not in first["data"]["results"][0]
    json.dumps(first, allow_nan=False)
    empty = search.invoke(
        arguments
        | {
            "created_from": "2027-01-01T00:00:00Z",
            "created_to": "2027-12-31T23:59:59Z",
        }
    )
    assert empty["data"]["results"] == []


def test_get_draft_returns_full_visible_artifact_and_hides_other_scope(tools: tuple) -> None:
    _search, get, visible, hidden = tools
    result = get.invoke({"artifact_id": visible.artifact_id})
    assert result["ok"] is True
    assert result["data"]["content"] == visible.content
    assert "conversation_id" not in result["data"]
    missing = get.invoke({"artifact_id": hidden.artifact_id})
    absent = get.invoke({"artifact_id": artifact_id_for("absent", ArtifactProducer.WRITER)})
    assert missing == absent == {
        "ok": False,
        "error": {
            "type": "artifact_not_found",
            "message": "The draft artifact was not found",
        },
    }


def test_invalid_and_dependency_failures_are_sanitized(tmp_path: Path) -> None:
    store = SQLiteArtifactStore(tmp_path / "artifacts.db", chunker=ParagraphChunker())
    store.initialize()
    value = artifact("visible", "conversation-1", "Visible draft")
    store.save_run((value,))
    retriever = HybridRetriever(
        store=store,
        embeddings=Embeddings(),
        reranker=Reranker(fail=True),
        fused_depth=10,
        rerank_depth=10,
    )
    search, _get = build_editorial_retrieval_tools(
        retriever=retriever, conversation_id="conversation-1"
    )
    invalid = search.invoke({"query": " ", "created_from": "not-a-date"})
    failed = search.invoke({"query": "visible", "rerank": True})
    assert invalid["error"]["type"] == "invalid_search_request"
    assert failed == invalid
    assert "private" not in json.dumps(failed)
    store.close()


def test_tools_execute_through_langchain_runnable_interface(tools: tuple) -> None:
    search, get, visible, _hidden = tools
    searched = search.invoke({"query": "visible", "rerank": False})
    loaded = get.invoke({"artifact_id": visible.artifact_id})
    assert searched["ok"] is True
    assert loaded["data"]["content"] == visible.content
