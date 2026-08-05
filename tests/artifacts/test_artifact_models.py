"""Artifact domain-model tests."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    artifact_id_for,
    content_sha256,
)

NOW = datetime(2026, 8, 5, tzinfo=UTC)


def artifact(producer: ArtifactProducer = ArtifactProducer.WRITER) -> EditorialArtifact:
    return EditorialArtifact(
        artifact_id=artifact_id_for("task-1", producer),
        task_id="task-1",
        producer=producer,
        created_at=NOW,
        conversation_id="conversation-1",
        user_request="Write a launch note",
        content="Launch draft.",
        content_sha256=content_sha256("Launch draft."),
    )


def test_writer_and_editor_are_the_only_supported_producers() -> None:
    assert artifact().producer is ArtifactProducer.WRITER
    assert artifact(ArtifactProducer.EDITOR).producer is ArtifactProducer.EDITOR
    with pytest.raises(ValueError, match="ArtifactProducer"):
        EditorialArtifact(
            artifact_id="artifact-1",
            task_id="task-1",
            producer="talker",  # type: ignore[arg-type]
            created_at=NOW,
            conversation_id="conversation-1",
            user_request="Request",
            content="Content",
            content_sha256=content_sha256("Content"),
        )


@pytest.mark.parametrize("field", ["artifact_id", "task_id", "user_request", "content"])
def test_artifact_rejects_blank_required_text(field: str) -> None:
    values = artifact().__dict__ | {field: " "}
    with pytest.raises(ValueError):
        EditorialArtifact(**values)


def test_artifact_requires_utc_and_matching_content_hash() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EditorialArtifact(**(artifact().__dict__ | {"created_at": datetime(2026, 8, 5)}))
    with pytest.raises(ValueError, match="match content"):
        EditorialArtifact(**(artifact().__dict__ | {"content_sha256": "0" * 64}))


def test_artifact_is_immutable_and_identity_is_task_scoped() -> None:
    value = artifact()
    with pytest.raises(FrozenInstanceError):
        value.content = "changed"  # type: ignore[misc]
    assert artifact_id_for("task-1", ArtifactProducer.WRITER) == value.artifact_id
    assert artifact_id_for("task-2", ArtifactProducer.WRITER) != value.artifact_id
    assert artifact_id_for("task-1", ArtifactProducer.EDITOR) != value.artifact_id
