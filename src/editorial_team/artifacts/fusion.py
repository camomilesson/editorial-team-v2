"""Reciprocal Rank Fusion for dense and lexical chunk rankings."""

from __future__ import annotations

from dataclasses import dataclass

from editorial_team.artifacts.lexical import LexicalCandidate
from editorial_team.artifacts.models import SearchableArtifactChunk
from editorial_team.artifacts.retrieval_types import DenseCandidate

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedCandidate:
    chunk: SearchableArtifactChunk
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_score: float


def reciprocal_rank_fusion(
    dense: tuple[DenseCandidate, ...],
    lexical: tuple[LexicalCandidate, ...],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    depth: int = 30,
) -> tuple[FusedCandidate, ...]:
    """Fuse unique chunks with 1-based ranks and deterministic ties."""

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    dense_by_id = {item.chunk.chunk.chunk_id: item for item in dense}
    lexical_by_id = {item.chunk.chunk.chunk_id: item for item in lexical}
    chunk_ids = set(dense_by_id) | set(lexical_by_id)
    fused: list[FusedCandidate] = []
    for chunk_id in chunk_ids:
        dense_item = dense_by_id.get(chunk_id)
        lexical_item = lexical_by_id.get(chunk_id)
        if dense_item is not None:
            chunk = dense_item.chunk
        elif lexical_item is not None:
            chunk = lexical_item.chunk
        else:  # pragma: no cover - constructed from the union above
            raise AssertionError("fused chunk is unavailable")
        score = 0.0
        if dense_item is not None:
            score += 1 / (rrf_k + dense_item.rank)
        if lexical_item is not None:
            score += 1 / (rrf_k + lexical_item.rank)
        fused.append(
            FusedCandidate(
                chunk=chunk,
                dense_rank=None if dense_item is None else dense_item.rank,
                dense_score=None if dense_item is None else dense_item.score,
                bm25_rank=None if lexical_item is None else lexical_item.rank,
                bm25_score=None if lexical_item is None else lexical_item.score,
                rrf_score=score,
            )
        )
    return tuple(
        sorted(
            fused,
            key=lambda item: (-item.rrf_score, item.chunk.chunk.chunk_id),
        )[:depth]
    )
