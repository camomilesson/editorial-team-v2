"""Synchronous SQLite persistence for heartbeat decisions."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from editorial_team.contracts.common import parse_utc_timestamp, timestamp_to_json
from editorial_team.contracts.identity import validate_identifier
from editorial_team.errors import DuplicateEntityError, EntityNotFoundError, ServiceError
from editorial_team.operations.models import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
    OperationalSnapshot,
)

MAX_RECENT_RESULTS = 1000


class HeartbeatStoreError(ServiceError):
    """A sanitized heartbeat persistence failure."""


class HeartbeatResultNotFoundError(HeartbeatStoreError, EntityNotFoundError):
    """The requested heartbeat result does not exist."""


class DuplicateHeartbeatResultError(HeartbeatStoreError, DuplicateEntityError):
    """A heartbeat result with the same ID already exists."""


class HeartbeatResultStore(Protocol):
    """Application-facing storage boundary for heartbeat decisions.

    This protocol is synchronous. Future async application code must place calls
    behind one explicit nonblocking boundary, such as ``asyncio.to_thread``.
    """

    def initialize(self) -> None: ...

    def save(self, result: HeartbeatResult) -> None: ...

    def get(self, result_id: str) -> HeartbeatResult: ...

    def list_recent(self, limit: int) -> tuple[HeartbeatResult, ...]: ...

    def mark_notification_sent(self, result_id: str) -> HeartbeatResult: ...


_COLUMNS = """
    result_id, observed_at, decision, reason_code, worker_running,
    queue_depth, queue_capacity, completed_jobs, failed_jobs,
    last_success_at, notification_sent
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS heartbeat_results (
    result_id TEXT PRIMARY KEY NOT NULL,
    observed_at TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('silence', 'notify')),
    reason_code TEXT NOT NULL CHECK (
        reason_code IN (
            'system_healthy', 'worker_stopped', 'repeated_failures', 'queue_pressure'
        )
    ),
    worker_running INTEGER NOT NULL CHECK (worker_running IN (0, 1)),
    queue_depth INTEGER NOT NULL CHECK (queue_depth >= 0),
    queue_capacity INTEGER NOT NULL CHECK (queue_capacity > 0),
    completed_jobs INTEGER NOT NULL CHECK (completed_jobs >= 0),
    failed_jobs INTEGER NOT NULL CHECK (failed_jobs >= 0),
    last_success_at TEXT,
    notification_sent INTEGER NOT NULL CHECK (notification_sent IN (0, 1)),
    CHECK (queue_depth <= queue_capacity),
    CHECK (
        (decision = 'silence' AND reason_code = 'system_healthy')
        OR
        (decision = 'notify' AND reason_code IN (
            'worker_stopped', 'repeated_failures', 'queue_pressure'
        ))
    ),
    CHECK (decision = 'notify' OR notification_sent = 0)
)
"""


class SQLiteHeartbeatResultStore:
    """Persist validated heartbeat results using short-lived SQLite connections."""

    def __init__(self, database_path: str | Path) -> None:
        if isinstance(database_path, str) and not database_path.strip():
            raise ValueError("database_path must not be blank")
        path = Path(database_path)
        self._database_path = path

    def initialize(self) -> None:
        """Create the parent directory and narrow table when absent."""

        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(_CREATE_TABLE)
        except (OSError, sqlite3.Error):
            raise HeartbeatStoreError("Heartbeat result store initialization failed") from None

    def save(self, result: HeartbeatResult) -> None:
        """Atomically insert a result without replacing an existing ID."""

        if not isinstance(result, HeartbeatResult):
            raise ValueError("result must be a HeartbeatResult")
        values = (
            result.id,
            timestamp_to_json(result.snapshot.observed_at),
            result.decision.value,
            result.reason_code.value,
            int(result.snapshot.worker_running),
            result.snapshot.queue_depth,
            result.snapshot.queue_capacity,
            result.snapshot.completed_jobs,
            result.snapshot.failed_jobs,
            (
                None
                if result.snapshot.last_success_at is None
                else timestamp_to_json(result.snapshot.last_success_at)
            ),
            int(result.notification_sent),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO heartbeat_results ({_COLUMNS}) VALUES ({','.join('?' * 11)})",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorcode in {
                sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
                sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            }:
                raise DuplicateHeartbeatResultError("Heartbeat result already exists") from None
            raise HeartbeatStoreError("Heartbeat result could not be saved") from None
        except sqlite3.Error:
            raise HeartbeatStoreError("Heartbeat result store operation failed") from None

    def get(self, result_id: str) -> HeartbeatResult:
        """Return one validated result or a sanitized not-found error."""

        result_id = validate_identifier(result_id, "result_id")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    f"SELECT {_COLUMNS} FROM heartbeat_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
        except sqlite3.Error:
            raise HeartbeatStoreError("Heartbeat result store operation failed") from None
        if row is None:
            raise HeartbeatResultNotFoundError("Heartbeat result was not found")
        return self._to_result(row)

    def list_recent(self, limit: int) -> tuple[HeartbeatResult, ...]:
        """Return newest results with result ID as the deterministic tie-breaker."""

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RECENT_RESULTS
        ):
            raise ValueError(f"limit must be between 1 and {MAX_RECENT_RESULTS}")
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT {_COLUMNS}
                    FROM heartbeat_results
                    ORDER BY observed_at DESC, result_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error:
            raise HeartbeatStoreError("Heartbeat result store operation failed") from None
        return tuple(self._to_result(row) for row in rows)

    def mark_notification_sent(self, result_id: str) -> HeartbeatResult:
        """Idempotently mark an existing NOTIFY result as delivered."""

        result_id = validate_identifier(result_id, "result_id")
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT decision FROM heartbeat_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                if row is None:
                    raise HeartbeatResultNotFoundError("Heartbeat result was not found")
                if row[0] != AdminDecision.NOTIFY.value:
                    raise HeartbeatStoreError("A SILENCE result cannot be marked as notified")
                connection.execute(
                    """
                    UPDATE heartbeat_results
                    SET notification_sent = 1
                    WHERE result_id = ?
                    """,
                    (result_id,),
                )
        except HeartbeatStoreError:
            raise
        except sqlite3.Error:
            raise HeartbeatStoreError("Heartbeat result store operation failed") from None
        return self.get(result_id)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    @staticmethod
    def _to_result(row: Sequence[object]) -> HeartbeatResult:
        try:
            last_success = (
                None if row[9] is None else parse_utc_timestamp(str(row[9]), "last_success_at")
            )
            return HeartbeatResult(
                id=str(row[0]),
                snapshot=OperationalSnapshot(
                    observed_at=parse_utc_timestamp(str(row[1]), "observed_at"),
                    worker_running=bool(row[4]),
                    queue_depth=int(row[5]),
                    queue_capacity=int(row[6]),
                    completed_jobs=int(row[7]),
                    failed_jobs=int(row[8]),
                    last_success_at=last_success,
                ),
                decision=AdminDecision(str(row[2])),
                reason_code=AdminReasonCode(str(row[3])),
                notification_sent=bool(row[10]),
            )
        except (ValueError, TypeError, IndexError):
            raise HeartbeatStoreError("Stored heartbeat result is invalid") from None
