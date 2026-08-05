"""Transactional SQLite artifact-store tests."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from editorial_team.artifacts import (
    ArtifactConflictError,
    ArtifactProducer,
    ArtifactStoreError,
    EditorialArtifact,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
    content_sha256,
)

NOW = datetime(2026, 8, 5, tzinfo=UTC)


def artifact(
    task_id: str,
    producer: ArtifactProducer,
    content: str,
    *,
    conversation_id: str = "conversation-1",
    created_at: datetime = NOW,
) -> EditorialArtifact:
    return EditorialArtifact(
        artifact_id=artifact_id_for(task_id, producer),
        task_id=task_id,
        producer=producer,
        created_at=created_at,
        conversation_id=conversation_id,
        user_request="Current request",
        content=content,
        content_sha256=content_sha256(content),
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteArtifactStore:
    value = SQLiteArtifactStore(
        tmp_path / "artifacts.db",
        chunker=ParagraphChunker(target_tokens=5, max_tokens=8, overlap_tokens=2),
    )
    value.initialize()
    yield value
    value.close()


def test_schema_round_trip_and_writer_only_transaction(store: SQLiteArtifactStore) -> None:
    writer = artifact("task-1", ArtifactProducer.WRITER, "One paragraph of writer content.")
    store.save_run((writer,))
    assert store.get_artifact(writer.artifact_id) == writer
    chunks = store.get_chunks(writer.artifact_id)
    assert chunks
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_writer_and_editor_transaction_and_queries(store: SQLiteArtifactStore) -> None:
    writer = artifact("task-1", ArtifactProducer.WRITER, "Writer output")
    editor = artifact("task-1", ArtifactProducer.EDITOR, "Editor output")
    other = artifact(
        "task-2",
        ArtifactProducer.WRITER,
        "Other output",
        conversation_id="conversation-2",
        created_at=NOW + timedelta(days=1),
    )
    store.save_run((writer, editor))
    store.save_run((other,))
    assert set(store.list_artifacts(conversation_id="conversation-1")) == {writer, editor}
    assert store.list_artifacts(created_from=NOW + timedelta(hours=1)) == (other,)


def test_identical_replay_is_idempotent_and_conflict_is_rejected(
    store: SQLiteArtifactStore,
) -> None:
    writer = artifact("task-1", ArtifactProducer.WRITER, "Writer output")
    store.save_run((writer,))
    store.save_run((writer,))
    assert store.list_artifacts() == (writer,)
    conflicting = replace(
        writer,
        content="Different output",
        content_sha256=content_sha256("Different output"),
    )
    with pytest.raises(ArtifactConflictError):
        store.save_run((conflicting,))
    assert store.get_artifact(writer.artifact_id) == writer


def test_partial_failure_rolls_back_complete_run(
    store: SQLiteArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = artifact("task-1", ArtifactProducer.WRITER, "Writer output")
    editor = artifact("task-1", ArtifactProducer.EDITOR, "Editor output")
    original = store._insert_or_verify  # type: ignore[attr-defined]
    calls = 0

    def fail_second(connection: sqlite3.Connection, item: EditorialArtifact, chunks: tuple) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.OperationalError("forced")
        original(connection, item, chunks)

    monkeypatch.setattr(store, "_insert_or_verify", fail_second)
    with pytest.raises(Exception, match="could not be saved"):
        store.save_run((writer, editor))
    assert store.list_artifacts() == ()


def test_foreign_keys_uniqueness_and_close(tmp_path: Path) -> None:
    path = tmp_path / "artifacts.db"
    store = SQLiteArtifactStore(path, chunker=ParagraphChunker())
    store.initialize()
    store.close()
    with pytest.raises(ArtifactStoreError, match="not initialized"):
        store.list_artifacts()
    with sqlite3.connect(path) as connection:
        index_names = {
            row[1] for row in connection.execute("PRAGMA index_list('artifacts')").fetchall()
        }
        assert "artifacts_task_id_idx" in index_names
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO artifact_chunks VALUES
                ('chunk-x', 'missing', 0, 'x', 0, 1, ?, 'writer',
                 'conversation-1', 'paragraph-heading-v1', ?)
                """,
                (NOW.isoformat(), content_sha256("x")),
            )
