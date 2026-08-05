"""Conversation-scoped local hybrid retrieval over stored artifact chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np

from editorial_team.artifacts.embeddings import EmbeddingModel
from editorial_team.artifacts.fusion import FusedCandidate, reciprocal_rank_fusion
from editorial_team.artifacts.lexical import BM25Retriever, LexicalCandidate
from editorial_team.artifacts.models import SearchableArtifactChunk
from editorial_team.artifacts.protocols import ArtifactStore
from editorial_team.artifacts.reranking import Reranker
from editorial_team.artifacts.retrieval_types import (
    DenseCandidate,
    RetrievedDraft,
    SearchRequest,
    SearchResult,
)
from editorial_team.errors import ServiceError

DEFAULT_DENSE_DEPTH = 30
DEFAULT_BM25_DEPTH = 30
DEFAULT_RRF_K = 60
DEFAULT_FUSED_DEPTH = 30
DEFAULT_RERANK_DEPTH = 15
MAX_EXCERPT_CHARACTERS = 600

VectorKey: TypeAlias = tuple[str, str, str]


class RetrievalError(ServiceError):
    """A sanitized dense, lexical, fusion, or reranking failure."""


@dataclass(frozen=True)
class RetrievalStages:
    """Exact ordered rankings retained for later evaluation."""

    dense: tuple[DenseCandidate, ...]
    bm25: tuple[LexicalCandidate, ...]
    fused: tuple[FusedCandidate, ...]
    results: tuple[SearchResult, ...]


class DenseRetriever:
    """Exact in-memory cosine search with a content-aware process cache."""

    def __init__(self, embeddings: EmbeddingModel, *, depth: int = DEFAULT_DENSE_DEPTH) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        self._embeddings = embeddings
        self.depth = depth
        self._cache: dict[VectorKey, np.ndarray] = {}

    def rank(
        self, query: str, chunks: Sequence[SearchableArtifactChunk]
    ) -> tuple[DenseCandidate, ...]:
        """Rank one already-filtered chunk sequence by exact cosine similarity."""

        if not chunks:
            return ()
        keys = [
            (item.chunk.chunk_id, item.chunk.content_sha256, self._embeddings.model_id)
            for item in chunks
        ]
        missing_indexes = [index for index, key in enumerate(keys) if key not in self._cache]
        if missing_indexes:
            texts = [chunks[index].chunk.content for index in missing_indexes]
            vectors = self._embeddings.embed_documents(texts)
            if len(vectors) != len(texts):
                raise RetrievalError("Embedding model returned an invalid result")
            for index, vector in zip(missing_indexes, vectors, strict=True):
                self._cache[keys[index]] = self._normalized(vector)
        query_vector = self._normalized(self._embeddings.embed_query(query))
        scored: list[tuple[SearchableArtifactChunk, float]] = []
        for item, key in zip(chunks, keys, strict=True):
            vector = self._cache[key]
            if vector.shape != query_vector.shape:
                raise RetrievalError("Embedding dimensions are inconsistent")
            scored.append((item, float(np.dot(query_vector, vector))))
        ordered = sorted(scored, key=lambda value: (-value[1], value[0].chunk.chunk_id))[
            : self.depth
        ]
        return tuple(
            DenseCandidate(chunk=item, rank=rank, score=score)
            for rank, (item, score) in enumerate(ordered, start=1)
        )

    @staticmethod
    def _normalized(vector: Sequence[float]) -> np.ndarray:
        try:
            value = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError):
            raise RetrievalError("Embedding model returned an invalid result") from None
        if value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
            raise RetrievalError("Embedding model returned an invalid result")
        norm = float(np.linalg.norm(value))
        if norm == 0:
            raise RetrievalError("Embedding model returned an invalid result")
        return value / norm


class HybridRetriever:
    """Dense + BM25 + RRF retrieval with optional cross-encoder reranking."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        embeddings: EmbeddingModel,
        reranker: Reranker,
        dense_depth: int = DEFAULT_DENSE_DEPTH,
        bm25_depth: int = DEFAULT_BM25_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
        fused_depth: int = DEFAULT_FUSED_DEPTH,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
    ) -> None:
        for name, value in (
            ("dense_depth", dense_depth),
            ("bm25_depth", bm25_depth),
            ("rrf_k", rrf_k),
            ("fused_depth", fused_depth),
            ("rerank_depth", rerank_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if rerank_depth > fused_depth:
            raise ValueError("rerank_depth must not exceed fused_depth")
        self._store = store
        self._dense = DenseRetriever(embeddings, depth=dense_depth)
        self._bm25 = BM25Retriever(depth=bm25_depth)
        self._reranker = reranker
        self.rrf_k = rrf_k
        self.fused_depth = fused_depth
        self.rerank_depth = rerank_depth

    def search(
        self,
        *,
        query: str,
        conversation_id: str,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        prefer_recent: bool = False,
        top_k: int = 5,
        rerank: bool = True,
    ) -> tuple[SearchResult, ...]:
        """Return the exact final ranking for one validated search request."""

        request = SearchRequest(
            query=query,
            conversation_id=conversation_id,
            created_from=created_from,
            created_to=created_to,
            prefer_recent=prefer_recent,
            top_k=top_k,
            rerank=rerank,
        )
        return self.search_with_stages(request).results

    def search_with_stages(self, request: SearchRequest) -> RetrievalStages:
        """Return final results together with untouched stage rankings."""

        if not isinstance(request, SearchRequest):
            raise ValueError("request must be a SearchRequest")
        chunks = self._store.list_searchable_chunks(
            conversation_id=request.conversation_id,
            created_from=request.created_from,
            created_to=request.created_to,
        )
        if not chunks:
            return RetrievalStages((), (), (), ())
        try:
            dense = self._dense.rank(request.query, chunks)
            bm25 = self._bm25.rank(request.query, chunks)
            fused = reciprocal_rank_fusion(
                dense,
                bm25,
                rrf_k=self.rrf_k,
                depth=self.fused_depth,
            )
            results = self._finalize(request, fused)
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("Hybrid retrieval failed") from None
        return RetrievalStages(dense, bm25, fused, results)

    def get_draft(self, *, artifact_id: str, conversation_id: str) -> RetrievedDraft | None:
        """Return complete content only inside the supplied conversation scope."""

        artifact = self._store.get_artifact_for_conversation(artifact_id, conversation_id)
        if artifact is None:
            return None
        return RetrievedDraft(
            artifact_id=artifact.artifact_id,
            task_id=artifact.task_id,
            producer=artifact.producer,
            created_at=artifact.created_at,
            conversation_id=artifact.conversation_id,
            user_request=artifact.user_request,
            content=artifact.content,
        )

    def _finalize(
        self, request: SearchRequest, fused: tuple[FusedCandidate, ...]
    ) -> tuple[SearchResult, ...]:
        candidates = fused[: self.rerank_depth] if request.rerank else fused
        rerank_scores: dict[str, float] = {}
        if request.rerank:
            passages = [item.chunk.chunk.content for item in candidates]
            scores = self._reranker.score(request.query, passages)
            if len(scores) != len(passages):
                raise RetrievalError("Reranker returned an invalid result")
            for item, score in zip(candidates, scores, strict=True):
                value = float(score)
                if not np.isfinite(value):
                    raise RetrievalError("Reranker returned an invalid result")
                rerank_scores[item.chunk.chunk.chunk_id] = value

        def order(item: FusedCandidate) -> tuple[float, float, float, str]:
            chunk = item.chunk.chunk
            active = rerank_scores.get(chunk.chunk_id, item.rrf_score)
            recency = -chunk.created_at.timestamp() if request.prefer_recent else 0.0
            return (-active, -item.rrf_score, recency, chunk.chunk_id)

        ordered = sorted(candidates, key=order)[: request.top_k]
        return tuple(
            self._result(index, item, rerank_scores.get(item.chunk.chunk.chunk_id))
            for index, item in enumerate(ordered, start=1)
        )

    @staticmethod
    def _result(
        rank: int, item: FusedCandidate, rerank_score: float | None
    ) -> SearchResult:
        stored = item.chunk
        chunk = stored.chunk
        excerpt = chunk.content
        if len(excerpt) > MAX_EXCERPT_CHARACTERS:
            excerpt = excerpt[: MAX_EXCERPT_CHARACTERS - 1].rstrip() + "…"
        return SearchResult(
            rank=rank,
            chunk_id=chunk.chunk_id,
            artifact_id=chunk.artifact_id,
            task_id=stored.task_id,
            excerpt=excerpt,
            created_at=chunk.created_at,
            producer=chunk.producer,
            dense_rank=item.dense_rank,
            dense_score=item.dense_score,
            bm25_rank=item.bm25_rank,
            bm25_score=item.bm25_score,
            rrf_score=item.rrf_score,
            rerank_score=rerank_score,
            chunk_ordinal=chunk.ordinal,
        )
