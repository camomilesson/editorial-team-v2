"""Immutable domain models for retrievable editorial artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier


class ArtifactProducer(StrEnum):
    """Editorial participants whose completed output enters the corpus."""

    WRITER = "writer"
    EDITOR = "editor"


def content_sha256(content: str) -> str:
    """Return the lowercase SHA-256 digest for exact UTF-8 content."""

    content = require_non_blank(content, "content")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def artifact_id_for(task_id: str, producer: ArtifactProducer) -> str:
    """Return a replay-stable artifact ID for one producer in one editorial run."""

    task_id = validate_identifier(task_id, "task_id")
    if not isinstance(producer, ArtifactProducer):
        raise ValueError("producer must be an ArtifactProducer")
    digest = hashlib.sha256(f"artifact:v1:{task_id}:{producer.value}".encode()).hexdigest()
    return f"artifact-v1-{digest}"


@dataclass(frozen=True)
class EditorialArtifact:
    """One complete immutable Writer or Editor output."""

    artifact_id: str
    task_id: str
    producer: ArtifactProducer
    created_at: datetime
    conversation_id: str
    user_request: str
    content: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", validate_identifier(self.artifact_id, "artifact_id")
        )
        object.__setattr__(self, "task_id", validate_identifier(self.task_id, "task_id"))
        if not isinstance(self.producer, ArtifactProducer):
            raise ValueError("producer must be an ArtifactProducer")
        if self.artifact_id != artifact_id_for(self.task_id, self.producer):
            raise ValueError("artifact_id must match task_id and producer")
        require_utc_timestamp(self.created_at, "created_at")
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(
            self, "user_request", require_non_blank(self.user_request, "user_request")
        )
        object.__setattr__(self, "content", require_non_blank(self.content, "content"))
        expected = content_sha256(self.content)
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 must match content")


@dataclass(frozen=True)
class ArtifactChunk:
    """One deterministic searchable slice of an editorial artifact."""

    chunk_id: str
    artifact_id: str
    ordinal: int
    content: str
    character_start: int
    character_end: int
    created_at: datetime
    producer: ArtifactProducer
    conversation_id: str
    chunker_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", validate_identifier(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self, "artifact_id", validate_identifier(self.artifact_id, "artifact_id")
        )
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        object.__setattr__(self, "content", require_non_blank(self.content, "content"))
        if (
            isinstance(self.character_start, bool)
            or not isinstance(self.character_start, int)
            or self.character_start < 0
        ):
            raise ValueError("character_start must be a nonnegative integer")
        if (
            isinstance(self.character_end, bool)
            or not isinstance(self.character_end, int)
            or self.character_end <= self.character_start
        ):
            raise ValueError("character_end must be greater than character_start")
        require_utc_timestamp(self.created_at, "created_at")
        if not isinstance(self.producer, ArtifactProducer):
            raise ValueError("producer must be an ArtifactProducer")
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        object.__setattr__(
            self,
            "chunker_version",
            validate_identifier(self.chunker_version, "chunker_version"),
        )
        expected = content_sha256(self.content)
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 must match content")
