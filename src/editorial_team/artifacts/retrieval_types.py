"""Immutable contracts shared by hybrid retrieval stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from editorial_team.artifacts.models import ArtifactProducer, SearchableArtifactChunk
from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier


@dataclass(frozen=True)
class SearchRequest:
    query: str
    conversation_id: str
    created_from: datetime | None = None
    created_to: datetime | None = None
    prefer_recent: bool = False
    top_k: int = 5
    rerank: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", require_non_blank(self.query, "query"))
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        for name in ("created_from", "created_to"):
            value = getattr(self, name)
            if value is not None:
                require_utc_timestamp(value, name)
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ValueError("created_from must not be later than created_to")
        if not isinstance(self.prefer_recent, bool):
            raise ValueError("prefer_recent must be a boolean")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= 10
        ):
            raise ValueError("top_k must be between 1 and 10")
        if not isinstance(self.rerank, bool):
            raise ValueError("rerank must be a boolean")


@dataclass(frozen=True)
class DenseCandidate:
    chunk: SearchableArtifactChunk
    rank: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: str
    artifact_id: str
    task_id: str
    excerpt: str
    created_at: datetime
    producer: ArtifactProducer
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None
    rrf_score: float
    rerank_score: float | None
    chunk_ordinal: int


@dataclass(frozen=True)
class RetrievedDraft:
    artifact_id: str
    task_id: str
    producer: ArtifactProducer
    created_at: datetime
    conversation_id: str
    user_request: str
    content: str
