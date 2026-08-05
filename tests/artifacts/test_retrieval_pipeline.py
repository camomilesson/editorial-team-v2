"""Deterministic tests for dense, lexical, fusion, and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    HybridRetriever,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
    content_sha256,
)
from editorial_team.artifacts.fusion import reciprocal_rank_fusion
from editorial_team.artifacts.lexical import BM25Retriever, LexicalCandidate, tokenize
from editorial_team.artifacts.retrieval import DenseRetriever
from editorial_team.artifacts.retrieval_types import DenseCandidate, SearchRequest

NOW = datetime(2026, 8, 5, 12, tzinfo=UTC)


@dataclass
class Embeddings:
    vectors: dict[str, list[float]]
    model_id: str = "fake-embeddings-v1"
    document_calls: list[tuple[str, ...]] = field(default_factory=list)
    query_calls: list[str] = field(default_factory=list)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vectors[text]


@dataclass
class Reranker:
    scores: dict[str, float]
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, tuple(passages)))
        return [self.scores[passage] for passage in passages]


def artifact(
    task_id: str,
    content: str,
    *,
    conversation_id: str = "conversation-1",
    created_at: datetime = NOW,
) -> EditorialArtifact:
    return EditorialArtifact(
        artifact_id=artifact_id_for(task_id, ArtifactProducer.WRITER),
        task_id=task_id,
        producer=ArtifactProducer.WRITER,
        created_at=created_at,
        conversation_id=conversation_id,
        user_request=f"Request for {task_id}",
        content=content,
        content_sha256=content_sha256(content),
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteArtifactStore:
    value = SQLiteArtifactStore(tmp_path / "artifacts.db", chunker=ParagraphChunker())
    value.initialize()
    yield value
    value.close()


def searchable(store: SQLiteArtifactStore, values: tuple[EditorialArtifact, ...]) -> tuple:
    for value in values:
        store.save_run((value,))
    return store.list_searchable_chunks(conversation_id="conversation-1")


def test_dense_exact_cosine_semantic_order_and_stable_tie(
    store: SQLiteArtifactStore,
) -> None:
    chunks = searchable(
        store,
        (
            artifact("task-car", "An automobile travels quickly."),
            artifact("task-food", "A recipe for tomato soup."),
            artifact("task-tie", "Another vehicle description."),
        ),
    )
    embeddings = Embeddings(
        {
            "fast car": [1, 0],
            "An automobile travels quickly.": [1, 0],
            "A recipe for tomato soup.": [0, 1],
            "Another vehicle description.": [1, 0],
        }
    )
    results = DenseRetriever(embeddings, depth=3).rank("fast car", chunks)
    assert results[0].score == pytest.approx(1.0)
    tied = [item.chunk.chunk.chunk_id for item in results[:2]]
    assert tied == sorted(tied)


def test_dense_filters_before_embedding_and_discovers_new_chunks(
    store: SQLiteArtifactStore,
) -> None:
    old = artifact("old", "Old eligible", created_at=NOW - timedelta(days=2))
    hidden = artifact("hidden", "Other conversation", conversation_id="conversation-2")
    store.save_run((old,))
    store.save_run((hidden,))
    embeddings = Embeddings(
        {
            "query": [1, 0],
            "Old eligible": [1, 0],
            "New eligible": [0.9, 0.1],
        }
    )
    dense = DenseRetriever(embeddings, depth=10)
    eligible = store.list_searchable_chunks(
        conversation_id="conversation-1", created_to=NOW - timedelta(days=1)
    )
    dense.rank("query", eligible)
    assert embeddings.document_calls == [("Old eligible",)]

    new = artifact("new", "New eligible", created_at=NOW)
    store.save_run((new,))
    all_chunks = store.list_searchable_chunks(conversation_id="conversation-1")
    dense.rank("query", all_chunks)
    assert embeddings.document_calls[-1] == ("New eligible",)


def test_bm25_tokenization_exact_term_acronym_and_ties(store: SQLiteArtifactStore) -> None:
    chunks = searchable(
        store,
        (
            artifact("rare", "Project ZX-81 launch notes"),
            artifact("common", "General launch notes"),
            artifact("tie", "Unrelated material"),
        ),
    )
    assert tokenize("Café, ZX-81 and O'Reilly!") == (
        "café",
        "zx-81",
        "and",
        "o'reilly",
    )
    results = BM25Retriever(depth=3).rank("ZX-81", chunks)
    assert results[0].chunk.task_id == "rare"
    assert results[0].score > results[1].score
    zero_ids = [item.chunk.chunk.chunk_id for item in results[1:]]
    assert zero_ids == sorted(zero_ids)


def test_rrf_formula_missing_stage_duplicate_removal_and_constant(
    store: SQLiteArtifactStore,
) -> None:
    first, second = searchable(
        store,
        (artifact("one", "One"), artifact("two", "Two")),
    )
    dense = (
        DenseCandidate(first, 1, 0.9),
        DenseCandidate(second, 2, 0.8),
    )
    lexical = (LexicalCandidate(first, 2, 1.2),)
    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=10, depth=10)
    assert len(fused) == 2
    assert fused[0].chunk == first
    assert fused[0].rrf_score == pytest.approx(1 / 11 + 1 / 12)
    assert fused[1].bm25_rank is None
    changed = reciprocal_rank_fusion(dense, lexical, rrf_k=20, depth=10)
    assert changed[0].rrf_score != fused[0].rrf_score


def test_hybrid_reranking_toggle_preserves_diagnostics_and_shortlist(
    store: SQLiteArtifactStore,
) -> None:
    values = (
        artifact("semantic", "Automobile speed guide"),
        artifact("lexical", "RareTerm reference"),
        artifact("better", "Preferred final passage"),
    )
    searchable(store, values)
    embeddings = Embeddings(
        {
            "fast car RareTerm": [1, 0],
            "Automobile speed guide": [1, 0],
            "RareTerm reference": [0.8, 0.2],
            "Preferred final passage": [0.7, 0.3],
        }
    )
    reranker = Reranker(
        {
            "Automobile speed guide": 0.2,
            "RareTerm reference": 0.4,
            "Preferred final passage": 0.9,
        }
    )
    retriever = HybridRetriever(
        store=store,
        embeddings=embeddings,
        reranker=reranker,
        dense_depth=3,
        bm25_depth=3,
        fused_depth=3,
        rerank_depth=2,
    )
    without = retriever.search(
        query="fast car RareTerm", conversation_id="conversation-1", top_k=2, rerank=False
    )
    assert not reranker.calls
    assert all(item.rerank_score is None for item in without)
    with_rerank = retriever.search(
        query="fast car RareTerm", conversation_id="conversation-1", top_k=2, rerank=True
    )
    assert len(reranker.calls[0][1]) == 2
    assert with_rerank[0].rerank_score >= with_rerank[1].rerank_score
    assert all(item.rrf_score > 0 for item in with_rerank)
    assert all(item.dense_rank is not None and item.bm25_rank is not None for item in with_rerank)


def test_recency_is_only_a_secondary_tie_break(store: SQLiteArtifactStore) -> None:
    older = artifact("older", "Older", created_at=NOW - timedelta(days=2))
    newer = artifact("newer", "Newer", created_at=NOW)
    searchable(store, (older, newer))
    embeddings = Embeddings({"Newer": [1, 0], "Older": [1, 0]})
    reranker = Reranker({"Older": 0.5, "Newer": 0.5})
    retriever = HybridRetriever(
        store=store,
        embeddings=embeddings,
        reranker=reranker,
        dense_depth=2,
        bm25_depth=2,
        fused_depth=2,
        rerank_depth=2,
    )
    recent = retriever.search(
        query="Newer",
        conversation_id="conversation-1",
        prefer_recent=True,
        top_k=2,
        rerank=True,
    )
    assert recent[0].task_id == "newer"
    reranker.scores = {"Older": 0.9, "Newer": 0.5}
    relevant = retriever.search(
        query="Newer",
        conversation_id="conversation-1",
        prefer_recent=True,
        top_k=2,
        rerank=True,
    )
    assert relevant[0].task_id == "older"


@pytest.mark.parametrize("top_k", [1, 3, 5, 10])
def test_end_to_end_top_k_empty_scope_dates_and_stage_inspection(
    store: SQLiteArtifactStore, top_k: int
) -> None:
    values = tuple(artifact(f"task-{index}", f"Passage {index}") for index in range(10))
    searchable(store, values)
    vectors = {"query": [1, 0]} | {
        value.content: [1, index / 100] for index, value in enumerate(values)
    }
    reranker = Reranker({value.content: float(index) for index, value in enumerate(values)})
    retriever = HybridRetriever(
        store=store,
        embeddings=Embeddings(vectors),
        reranker=reranker,
        dense_depth=30,
        bm25_depth=30,
        fused_depth=30,
        rerank_depth=15,
    )
    stages = retriever.search_with_stages(
        SearchRequest("query", "conversation-1", top_k=top_k, rerank=False)
    )
    assert len(stages.results) == top_k
    assert [item.rank for item in stages.results] == list(range(1, top_k + 1))
    assert stages.dense and stages.bm25 and stages.fused
    assert retriever.search(query="query", conversation_id="empty", rerank=False) == ()
    with pytest.raises(ValueError, match="created_from"):
        retriever.search(
            query="query",
            conversation_id="conversation-1",
            created_from=NOW,
            created_to=NOW - timedelta(days=1),
        )
