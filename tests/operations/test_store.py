"""Tests for narrow SQLite heartbeat-result persistence."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from editorial_team.operations import (
    MAX_RECENT_RESULTS,
    AdminDecision,
    AdminReasonCode,
    DuplicateHeartbeatResultError,
    HeartbeatResult,
    HeartbeatResultNotFoundError,
    HeartbeatStoreError,
    OperationalSnapshot,
    SQLiteHeartbeatResultStore,
)

OBSERVED = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)
EXPECTED_COLUMNS = {
    "result_id",
    "observed_at",
    "decision",
    "reason_code",
    "worker_running",
    "queue_depth",
    "queue_capacity",
    "completed_jobs",
    "failed_jobs",
    "last_success_at",
    "notification_sent",
}


def make_result(
    result_id: str = "heartbeat-1",
    *,
    observed_at: datetime = OBSERVED,
    decision: AdminDecision = AdminDecision.SILENCE,
    reason: AdminReasonCode = AdminReasonCode.SYSTEM_HEALTHY,
    last_success_at: datetime | None = OBSERVED - timedelta(minutes=2),
    notification_sent: bool = False,
) -> HeartbeatResult:
    return HeartbeatResult(
        id=result_id,
        snapshot=OperationalSnapshot(
            observed_at=observed_at,
            worker_running=decision is AdminDecision.SILENCE,
            queue_depth=2,
            queue_capacity=100,
            completed_jobs=5,
            failed_jobs=1,
            last_success_at=last_success_at,
        ),
        decision=decision,
        reason_code=reason,
        notification_sent=notification_sent,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteHeartbeatResultStore:
    repository = SQLiteHeartbeatResultStore(tmp_path / "data" / "heartbeats.db")
    repository.initialize()
    return repository


def test_initialize_creates_parent_database_and_table(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "heartbeats.db"
    store = SQLiteHeartbeatResultStore(database)

    assert not database.parent.exists()
    store.initialize()

    assert database.is_file()
    with sqlite3.connect(database) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert tables == [("heartbeat_results",)]


def test_initialize_is_idempotent_and_preserves_rows(tmp_path: Path) -> None:
    store = SQLiteHeartbeatResultStore(tmp_path / "heartbeats.db")
    store.initialize()
    store.save(make_result())

    store.initialize()

    assert store.get("heartbeat-1") == make_result()


@pytest.mark.parametrize(
    "result",
    [
        make_result(),
        make_result(
            decision=AdminDecision.NOTIFY,
            reason=AdminReasonCode.WORKER_STOPPED,
        ),
        make_result(last_success_at=None),
    ],
)
def test_save_and_get_round_trip(
    store: SQLiteHeartbeatResultStore, result: HeartbeatResult
) -> None:
    store.save(result)

    assert store.get(result.id) == result


def test_duplicate_is_sanitized_and_does_not_replace(
    store: SQLiteHeartbeatResultStore,
) -> None:
    first = make_result()
    store.save(first)

    with pytest.raises(DuplicateHeartbeatResultError) as error:
        store.save(
            make_result(
                decision=AdminDecision.NOTIFY,
                reason=AdminReasonCode.QUEUE_PRESSURE,
            )
        )

    assert str(error.value) == "Heartbeat result already exists"
    assert store.get(first.id) == first


def test_missing_result_uses_sanitized_error(
    store: SQLiteHeartbeatResultStore,
) -> None:
    with pytest.raises(HeartbeatResultNotFoundError) as error:
        store.get("missing")

    assert str(error.value) == "Heartbeat result was not found"


def test_recent_results_are_newest_first(store: SQLiteHeartbeatResultStore) -> None:
    for offset in range(3):
        store.save(
            make_result(
                f"heartbeat-{offset}",
                observed_at=OBSERVED + timedelta(minutes=offset),
            )
        )

    assert [result.id for result in store.list_recent(2)] == [
        "heartbeat-2",
        "heartbeat-1",
    ]


def test_equal_timestamps_use_descending_id_order(
    store: SQLiteHeartbeatResultStore,
) -> None:
    for result_id in ("heartbeat-a", "heartbeat-c", "heartbeat-b"):
        store.save(make_result(result_id))

    assert [result.id for result in store.list_recent(3)] == [
        "heartbeat-c",
        "heartbeat-b",
        "heartbeat-a",
    ]


@pytest.mark.parametrize("limit", [0, -1, MAX_RECENT_RESULTS + 1, True, 1.5])
def test_invalid_recent_limit_is_rejected(
    store: SQLiteHeartbeatResultStore, limit: object
) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        store.list_recent(limit)  # type: ignore[arg-type]


def test_results_survive_a_new_store_instance(tmp_path: Path) -> None:
    database = tmp_path / "heartbeats.db"
    first_store = SQLiteHeartbeatResultStore(database)
    first_store.initialize()
    first_store.save(make_result())

    second_store = SQLiteHeartbeatResultStore(database)

    assert second_store.get("heartbeat-1") == make_result()


def test_notify_can_be_marked_sent_idempotently(
    store: SQLiteHeartbeatResultStore,
) -> None:
    result = make_result(
        decision=AdminDecision.NOTIFY,
        reason=AdminReasonCode.REPEATED_FAILURES,
    )
    store.save(result)

    updated = store.mark_notification_sent(result.id)
    repeated = store.mark_notification_sent(result.id)

    assert updated.notification_sent is True
    assert repeated == updated
    assert repeated.snapshot == result.snapshot
    assert repeated.decision == result.decision


def test_silence_cannot_be_marked_sent(store: SQLiteHeartbeatResultStore) -> None:
    store.save(make_result())

    with pytest.raises(HeartbeatStoreError, match="SILENCE"):
        store.mark_notification_sent("heartbeat-1")

    assert store.get("heartbeat-1").notification_sent is False


def test_marking_missing_result_fails_cleanly(
    store: SQLiteHeartbeatResultStore,
) -> None:
    with pytest.raises(HeartbeatResultNotFoundError):
        store.mark_notification_sent("missing")


def test_database_exception_is_sanitized(tmp_path: Path) -> None:
    database_directory = tmp_path / "not-a-database"
    database_directory.mkdir()
    store = SQLiteHeartbeatResultStore(database_directory)

    with pytest.raises(HeartbeatStoreError) as error:
        store.initialize()

    assert str(error.value) == "Heartbeat result store initialization failed"
    assert str(database_directory) not in str(error.value)


def test_parent_is_created_only_by_initialize(tmp_path: Path) -> None:
    parent = tmp_path / "not-created-yet"
    store = SQLiteHeartbeatResultStore(parent / "heartbeats.db")

    assert not parent.exists()
    with pytest.raises(HeartbeatStoreError):
        store.save(make_result())
    assert not parent.exists()

    store.initialize()
    assert parent.is_dir()


def test_schema_has_only_operational_columns(tmp_path: Path) -> None:
    database = tmp_path / "heartbeats.db"
    SQLiteHeartbeatResultStore(database).initialize()

    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(heartbeat_results)"
            ).fetchall()
        }

    assert columns == EXPECTED_COLUMNS
    forbidden_fragments = {
        "message",
        "prompt",
        "draft",
        "chat",
        "username",
        "user",
        "model",
    }
    assert not any(
        fragment in column for column in columns for fragment in forbidden_fragments
    )


def test_database_constraints_reject_invalid_primitive_rows(tmp_path: Path) -> None:
    database = tmp_path / "heartbeats.db"
    SQLiteHeartbeatResultStore(database).initialize()
    valid = (
        "direct-1",
        "2026-07-29T08:00:00Z",
        "silence",
        "system_healthy",
        1,
        0,
        100,
        1,
        0,
        None,
        0,
    )
    cases = [
        (*valid[:4], 2, *valid[5:]),
        (*valid[:5], -1, *valid[6:]),
        (*valid[:6], 0, *valid[7:]),
        (*valid[:7], -1, *valid[8:]),
        (*valid[:8], -1, *valid[9:]),
        (*valid[:2], "unknown", *valid[3:]),
        (*valid[:3], "unknown", *valid[4:]),
        (*valid[:2], "silence", "worker_stopped", *valid[4:]),
        (*valid[:-1], 1),
    ]

    with sqlite3.connect(database) as connection:
        for index, values in enumerate(cases):
            values = (f"direct-{index}", *values[1:])
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO heartbeat_results (
                        result_id, observed_at, decision, reason_code,
                        worker_running, queue_depth, queue_capacity,
                        completed_jobs, failed_jobs, last_success_at,
                        notification_sent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )


def test_repository_tests_do_not_create_runtime_database() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert not (repository_root / "runtime_data").exists()
