"""Transactional SQLite persistence for complete editorial artifacts."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Final

from editorial_team.artifacts.chunking import ParagraphChunker
from editorial_team.artifacts.models import ArtifactChunk, ArtifactProducer, EditorialArtifact
from editorial_team.contracts.common import parse_utc_timestamp, timestamp_to_json
from editorial_team.contracts.identity import validate_identifier
from editorial_team.errors import DuplicateEntityError, EntityNotFoundError, ServiceError


class ArtifactStoreError(ServiceError):
    """A sanitized artifact persistence failure."""


class ArtifactConflictError(ArtifactStoreError, DuplicateEntityError):
    """A replay conflicts with immutable stored artifact data."""


class ArtifactNotFoundError(ArtifactStoreError, EntityNotFoundError):
    """A requested artifact does not exist."""


_CREATE_ARTIFACTS: Final = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY NOT NULL,
    task_id TEXT NOT NULL,
    producer TEXT NOT NULL CHECK (producer IN ('writer', 'editor')),
    created_at TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    user_request TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    UNIQUE (task_id, producer)
)
"""

_CREATE_CHUNKS: Final = """
CREATE TABLE IF NOT EXISTS artifact_chunks (
    chunk_id TEXT PRIMARY KEY NOT NULL,
    artifact_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    content TEXT NOT NULL,
    character_start INTEGER NOT NULL CHECK (character_start >= 0),
    character_end INTEGER NOT NULL CHECK (character_end > character_start),
    created_at TEXT NOT NULL,
    producer TEXT NOT NULL CHECK (producer IN ('writer', 'editor')),
    conversation_id TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
    UNIQUE (artifact_id, ordinal)
)
"""

_INDEXES: Final = (
    "CREATE INDEX IF NOT EXISTS artifacts_task_id_idx ON artifacts(task_id)",
    "CREATE INDEX IF NOT EXISTS artifacts_created_at_idx ON artifacts(created_at)",
    "CREATE INDEX IF NOT EXISTS artifacts_conversation_id_idx ON artifacts(conversation_id)",
    "CREATE INDEX IF NOT EXISTS chunks_artifact_id_idx ON artifact_chunks(artifact_id)",
    "CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON artifact_chunks(created_at)",
    "CREATE INDEX IF NOT EXISTS chunks_conversation_id_idx ON artifact_chunks(conversation_id)",
    "CREATE INDEX IF NOT EXISTS chunks_artifact_ordinal_idx "
    "ON artifact_chunks(artifact_id, ordinal)",
)


class SQLiteArtifactStore:
    """Store each completed editorial run in one all-or-nothing transaction."""

    def __init__(self, database_path: str | Path, *, chunker: ParagraphChunker) -> None:
        if isinstance(database_path, str) and not database_path.strip():
            raise ValueError("database_path must not be blank")
        if not isinstance(chunker, ParagraphChunker):
            raise ValueError("chunker must be a ParagraphChunker")
        self._database_path = Path(database_path)
        self._chunker = chunker
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Open the database and initialize its narrow schema exactly once."""

        if self._connection is not None:
            return
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._database_path, check_same_thread=False)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(_CREATE_ARTIFACTS)
            connection.execute(_CREATE_CHUNKS)
            for statement in _INDEXES:
                connection.execute(statement)
            connection.commit()
        except (OSError, sqlite3.Error):
            if "connection" in locals():
                connection.close()
            raise ArtifactStoreError("Artifact store initialization failed") from None
        self._connection = connection

    def save_run(self, artifacts: tuple[EditorialArtifact, ...]) -> None:
        """Atomically save the exact immutable output set for one completed run."""

        self._validate_run(artifacts)
        derived = tuple((artifact, self._chunker.chunk(artifact)) for artifact in artifacts)
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            task_id = artifacts[0].task_id
            stored_rows = connection.execute(
                "SELECT artifact_id, producer FROM artifacts WHERE task_id = ?",
                (task_id,),
            ).fetchall()
            stored_producers = {row[1] for row in stored_rows}
            incoming_producers = {artifact.producer.value for artifact in artifacts}
            if stored_producers and stored_producers != incoming_producers:
                raise ArtifactConflictError("Stored editorial run conflicts with replay")
            for artifact, chunks in derived:
                self._insert_or_verify(connection, artifact, chunks)
            connection.commit()
        except ArtifactConflictError:
            connection.rollback()
            raise
        except (sqlite3.Error, ValueError):
            connection.rollback()
            raise ArtifactStoreError("Editorial artifacts could not be saved") from None

    def get_artifact(self, artifact_id: str) -> EditorialArtifact:
        """Load one complete artifact by ID."""

        artifact_id = validate_identifier(artifact_id, "artifact_id")
        try:
            row = (
                self._require_connection()
                .execute(
                    """
                SELECT artifact_id, task_id, producer, created_at, conversation_id,
                       user_request, content, content_sha256
                FROM artifacts WHERE artifact_id = ?
                """,
                    (artifact_id,),
                )
                .fetchone()
            )
        except sqlite3.Error:
            raise ArtifactStoreError("Artifact store operation failed") from None
        if row is None:
            raise ArtifactNotFoundError("Artifact was not found")
        return self._artifact_from_row(row)

    def get_chunks(self, artifact_id: str) -> tuple[ArtifactChunk, ...]:
        """Load all chunks for an artifact in canonical ordinal order."""

        artifact_id = validate_identifier(artifact_id, "artifact_id")
        try:
            rows = (
                self._require_connection()
                .execute(
                    """
                SELECT chunk_id, artifact_id, ordinal, content, character_start,
                       character_end, created_at, producer, conversation_id,
                       chunker_version, content_sha256
                FROM artifact_chunks WHERE artifact_id = ? ORDER BY ordinal
                """,
                    (artifact_id,),
                )
                .fetchall()
            )
        except sqlite3.Error:
            raise ArtifactStoreError("Artifact store operation failed") from None
        return tuple(self._chunk_from_row(row) for row in rows)

    def list_artifacts(
        self,
        *,
        conversation_id: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> tuple[EditorialArtifact, ...]:
        """List artifacts by factual conversation and UTC chronology filters."""

        clauses: list[str] = []
        values: list[str] = []
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            values.append(validate_identifier(conversation_id, "conversation_id"))
        for _name, value, operator in (
            ("created_from", created_from, ">="),
            ("created_to", created_to, "<="),
        ):
            if value is not None:
                clauses.append(f"created_at {operator} ?")
                values.append(timestamp_to_json(value))
        if created_from is not None and created_to is not None and created_from > created_to:
            raise ValueError("created_from must not be later than created_to")
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        try:
            rows = (
                self._require_connection()
                .execute(
                    """
                SELECT artifact_id, task_id, producer, created_at, conversation_id,
                       user_request, content, content_sha256
                FROM artifacts
                """
                    + where
                    + " ORDER BY created_at, artifact_id",
                    values,
                )
                .fetchall()
            )
        except sqlite3.Error:
            raise ArtifactStoreError("Artifact store operation failed") from None
        return tuple(self._artifact_from_row(row) for row in rows)

    def close(self) -> None:
        """Close the owned database connection once."""

        connection = self._connection
        if connection is not None:
            self._connection = None
            connection.close()

    def _insert_or_verify(
        self,
        connection: sqlite3.Connection,
        artifact: EditorialArtifact,
        chunks: tuple[ArtifactChunk, ...],
    ) -> None:
        row = connection.execute(
            """
            SELECT artifact_id, task_id, producer, created_at, conversation_id,
                   user_request, content, content_sha256, chunker_version
            FROM artifacts WHERE task_id = ? AND producer = ?
            """,
            (artifact.task_id, artifact.producer.value),
        ).fetchone()
        expected = self._artifact_values(artifact)
        if row is not None:
            if tuple(row) != expected:
                raise ArtifactConflictError("Stored artifact conflicts with replay")
            existing_chunks = connection.execute(
                """
                SELECT chunk_id, artifact_id, ordinal, content, character_start,
                       character_end, created_at, producer, conversation_id,
                       chunker_version, content_sha256
                FROM artifact_chunks WHERE artifact_id = ? ORDER BY ordinal
                """,
                (artifact.artifact_id,),
            ).fetchall()
            if tuple(tuple(item) for item in existing_chunks) != tuple(
                self._chunk_values(chunk) for chunk in chunks
            ):
                raise ArtifactConflictError("Stored artifact chunks conflict with replay")
            return
        connection.execute(
            """
            INSERT INTO artifacts (
                artifact_id, task_id, producer, created_at, conversation_id,
                user_request, content, content_sha256, chunker_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            expected,
        )
        connection.executemany(
            """
            INSERT INTO artifact_chunks (
                chunk_id, artifact_id, ordinal, content, character_start,
                character_end, created_at, producer, conversation_id,
                chunker_version, content_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(self._chunk_values(chunk) for chunk in chunks),
        )

    def _artifact_values(self, artifact: EditorialArtifact) -> tuple[object, ...]:
        return (
            artifact.artifact_id,
            artifact.task_id,
            artifact.producer.value,
            timestamp_to_json(artifact.created_at),
            artifact.conversation_id,
            artifact.user_request,
            artifact.content,
            artifact.content_sha256,
            self._chunker.version,
        )

    @staticmethod
    def _chunk_values(chunk: ArtifactChunk) -> tuple[object, ...]:
        return (
            chunk.chunk_id,
            chunk.artifact_id,
            chunk.ordinal,
            chunk.content,
            chunk.character_start,
            chunk.character_end,
            timestamp_to_json(chunk.created_at),
            chunk.producer.value,
            chunk.conversation_id,
            chunk.chunker_version,
            chunk.content_sha256,
        )

    @staticmethod
    def _validate_run(artifacts: tuple[EditorialArtifact, ...]) -> None:
        if (
            not isinstance(artifacts, tuple)
            or not artifacts
            or not all(isinstance(artifact, EditorialArtifact) for artifact in artifacts)
        ):
            raise ValueError("artifacts must be a non-empty tuple of EditorialArtifact values")
        if len(artifacts) not in {1, 2}:
            raise ValueError("an editorial run must contain one or two artifacts")
        task_ids = {artifact.task_id for artifact in artifacts}
        if len(task_ids) != 1:
            raise ValueError("artifacts must share one task_id")
        provenance = {
            (artifact.created_at, artifact.conversation_id, artifact.user_request)
            for artifact in artifacts
        }
        if len(provenance) != 1:
            raise ValueError("artifacts in one run must share factual provenance")
        producers = tuple(artifact.producer for artifact in artifacts)
        if producers not in {
            (ArtifactProducer.WRITER,),
            (ArtifactProducer.WRITER, ArtifactProducer.EDITOR),
        }:
            raise ValueError("artifacts must contain Writer and optional Editor output in order")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ArtifactStoreError("Artifact store is not initialized")
        return self._connection

    @staticmethod
    def _artifact_from_row(row: tuple[object, ...]) -> EditorialArtifact:
        try:
            return EditorialArtifact(
                artifact_id=str(row[0]),
                task_id=str(row[1]),
                producer=ArtifactProducer(str(row[2])),
                created_at=parse_utc_timestamp(str(row[3]), "created_at"),
                conversation_id=str(row[4]),
                user_request=str(row[5]),
                content=str(row[6]),
                content_sha256=str(row[7]),
            )
        except (ValueError, TypeError, IndexError):
            raise ArtifactStoreError("Stored artifact is invalid") from None

    @staticmethod
    def _chunk_from_row(row: tuple[object, ...]) -> ArtifactChunk:
        try:
            return ArtifactChunk(
                chunk_id=str(row[0]),
                artifact_id=str(row[1]),
                ordinal=int(row[2]),
                content=str(row[3]),
                character_start=int(row[4]),
                character_end=int(row[5]),
                created_at=parse_utc_timestamp(str(row[6]), "created_at"),
                producer=ArtifactProducer(str(row[7])),
                conversation_id=str(row[8]),
                chunker_version=str(row[9]),
                content_sha256=str(row[10]),
            )
        except (ValueError, TypeError, IndexError):
            raise ArtifactStoreError("Stored artifact chunk is invalid") from None
