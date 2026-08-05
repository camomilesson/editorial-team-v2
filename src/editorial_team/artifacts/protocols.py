"""Application-facing artifact persistence contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from editorial_team.artifacts.models import (
    ArtifactChunk,
    EditorialArtifact,
    SearchableArtifactChunk,
)


class ArtifactStore(Protocol):
    """Persist and query complete editorial runs."""

    def initialize(self) -> None: ...

    def save_run(self, artifacts: tuple[EditorialArtifact, ...]) -> None: ...

    def get_artifact(self, artifact_id: str) -> EditorialArtifact: ...

    def get_artifact_for_conversation(
        self, artifact_id: str, conversation_id: str
    ) -> EditorialArtifact | None: ...

    def get_chunks(self, artifact_id: str) -> tuple[ArtifactChunk, ...]: ...

    def list_searchable_chunks(
        self,
        *,
        conversation_id: str,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[SearchableArtifactChunk, ...]: ...

    def list_artifacts(
        self,
        *,
        conversation_id: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[EditorialArtifact, ...]: ...

    def close(self) -> None: ...
