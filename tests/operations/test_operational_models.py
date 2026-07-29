"""Tests for content-free operational heartbeat contracts."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from editorial_team.operations import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
    OperationalSnapshot,
)

OBSERVED = datetime(2026, 7, 29, 8, 0, tzinfo=UTC)


def snapshot(**changes: object) -> OperationalSnapshot:
    values = {
        "observed_at": OBSERVED,
        "worker_running": True,
        "queue_depth": 1,
        "queue_capacity": 100,
        "completed_jobs": 4,
        "failed_jobs": 0,
        "last_success_at": OBSERVED - timedelta(minutes=1),
    }
    values.update(changes)
    return OperationalSnapshot(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (AdminDecision.SILENCE, AdminReasonCode.SYSTEM_HEALTHY),
        (AdminDecision.NOTIFY, AdminReasonCode.WORKER_STOPPED),
        (AdminDecision.NOTIFY, AdminReasonCode.REPEATED_FAILURES),
        (AdminDecision.NOTIFY, AdminReasonCode.QUEUE_PRESSURE),
    ],
)
def test_valid_decision_reason_combinations(
    decision: AdminDecision, reason: AdminReasonCode
) -> None:
    result = HeartbeatResult("heartbeat-1", snapshot(), decision, reason)

    assert result.decision is decision
    assert result.reason_code is reason


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (AdminDecision.SILENCE, AdminReasonCode.WORKER_STOPPED),
        (AdminDecision.SILENCE, AdminReasonCode.REPEATED_FAILURES),
        (AdminDecision.SILENCE, AdminReasonCode.QUEUE_PRESSURE),
        (AdminDecision.NOTIFY, AdminReasonCode.SYSTEM_HEALTHY),
    ],
)
def test_invalid_decision_reason_combinations_are_rejected(
    decision: AdminDecision, reason: AdminReasonCode
) -> None:
    with pytest.raises(ValueError):
        HeartbeatResult("heartbeat-1", snapshot(), decision, reason)


def test_silence_cannot_be_marked_as_notified() -> None:
    with pytest.raises(ValueError, match="cannot be marked"):
        HeartbeatResult(
            "heartbeat-1",
            snapshot(),
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
            notification_sent=True,
        )


@pytest.mark.parametrize("field", ["queue_depth", "completed_jobs", "failed_jobs"])
def test_negative_counters_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        snapshot(**{field: -1})


@pytest.mark.parametrize("capacity", [0, -1])
def test_nonpositive_capacity_is_rejected(capacity: int) -> None:
    with pytest.raises(ValueError):
        snapshot(queue_capacity=capacity)


def test_depth_cannot_exceed_capacity() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        snapshot(queue_depth=2, queue_capacity=1)


@pytest.mark.parametrize("field", ["observed_at", "last_success_at"])
def test_naive_timestamps_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        snapshot(**{field: datetime(2026, 7, 29, 8, 0)})


def test_last_success_cannot_follow_observation() -> None:
    with pytest.raises(ValueError, match="must not be after"):
        snapshot(last_success_at=OBSERVED + timedelta(seconds=1))


def test_blank_result_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        HeartbeatResult(
            " ",
            snapshot(),
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
        )


def test_models_are_immutable() -> None:
    result = HeartbeatResult(
        "heartbeat-1",
        snapshot(),
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
    )

    with pytest.raises(FrozenInstanceError):
        result.notification_sent = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.snapshot.queue_depth = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_running", 1),
        ("queue_depth", True),
        ("queue_capacity", 1.5),
        ("completed_jobs", "4"),
        ("failed_jobs", False),
    ],
)
def test_snapshot_requires_exact_primitive_types(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        snapshot(**{field: value})
