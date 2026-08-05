"""Manual hybrid-search command tests with deterministic fakes."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO

from editorial_team.artifacts import ArtifactProducer
from editorial_team.artifacts.retrieval_types import SearchResult
from scripts.search_artifact_corpus import build_parser, execute_search


class Retriever:
    def search(self, **kwargs: object) -> tuple[SearchResult, ...]:
        assert kwargs["conversation_id"] == "conversation-1"
        assert kwargs["rerank"] is False
        return (
            SearchResult(
                rank=1,
                chunk_id="chunk-1",
                artifact_id="artifact-1",
                task_id="task-1",
                excerpt="Matching excerpt",
                created_at=datetime(2026, 8, 5, tzinfo=UTC),
                producer=ArtifactProducer.WRITER,
                dense_rank=2,
                dense_score=0.8,
                bm25_rank=1,
                bm25_score=4.2,
                rrf_score=0.032,
                rerank_score=None,
                chunk_ordinal=0,
            ),
        )


def test_parser_supports_required_search_options() -> None:
    arguments = build_parser().parse_args(
        [
            "launch draft",
            "--conversation-id",
            "conversation-1",
            "--created-from",
            "2026-08-01T00:00:00Z",
            "--top-k",
            "3",
            "--no-rerank",
            "--prefer-recent",
            "--database",
            "/tmp/artifacts.db",
        ]
    )
    assert arguments.query == "launch draft"
    assert arguments.top_k == 3
    assert arguments.rerank is False
    assert arguments.prefer_recent is True


def test_execute_search_prints_exact_ranks_and_diagnostics() -> None:
    stream = StringIO()
    results = execute_search(
        Retriever(),  # type: ignore[arg-type]
        query="launch draft",
        conversation_id="conversation-1",
        created_from=None,
        created_to=None,
        top_k=5,
        rerank=False,
        prefer_recent=False,
        stream=stream,
    )
    output = stream.getvalue()
    assert len(results) == 1
    assert "#1 artifact=artifact-1 chunk=chunk-1" in output
    assert "dense_rank=2 bm25_rank=1" in output
    assert "rrf=0.03200000 reranker=none" in output
    assert "Matching excerpt" in output
