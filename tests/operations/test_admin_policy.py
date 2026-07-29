"""Tests for immutable deterministic Admin policy."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from editorial_team.operations import (
    AdminDecision,
    AdminPolicy,
    AdminReasonCode,
    OperationalSnapshot,
    expected_admin_assessment,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def snapshot(**changes: object) -> OperationalSnapshot:
    values = {
        "observed_at": NOW,
        "worker_running": True,
        "queue_depth": 0,
        "queue_capacity": 100,
        "completed_jobs": 1,
        "failed_jobs": 0,
        "last_success_at": NOW,
    }
    values.update(changes)
    return OperationalSnapshot(**values)  # type: ignore[arg-type]


def test_policy_defaults_and_custom_values() -> None:
    assert AdminPolicy() == AdminPolicy(failure_threshold=3, queue_pressure_ratio=0.8)
    assert AdminPolicy(5, 0.6).failure_threshold == 5
    assert AdminPolicy(5, 0.6).queue_pressure_ratio == 0.6


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5])
def test_invalid_failure_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(ValueError):
        AdminPolicy(failure_threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01, True, float("nan"), float("inf")])
def test_invalid_queue_ratio_is_rejected(ratio: object) -> None:
    with pytest.raises(ValueError):
        AdminPolicy(queue_pressure_ratio=ratio)  # type: ignore[arg-type]


def test_policy_is_immutable() -> None:
    policy = AdminPolicy()

    with pytest.raises(FrozenInstanceError):
        policy.failure_threshold = 9  # type: ignore[misc]


def test_assessment_is_immutable() -> None:
    assessment = expected_admin_assessment(snapshot(), AdminPolicy())

    with pytest.raises(FrozenInstanceError):
        assessment.decision = AdminDecision.NOTIFY  # type: ignore[misc]


@pytest.mark.parametrize(
    ("runtime_snapshot", "decision", "reason"),
    [
        (
            snapshot(),
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
        ),
        (
            snapshot(worker_running=False),
            AdminDecision.NOTIFY,
            AdminReasonCode.WORKER_STOPPED,
        ),
        (
            snapshot(failed_jobs=2),
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
        ),
        (
            snapshot(failed_jobs=3),
            AdminDecision.NOTIFY,
            AdminReasonCode.REPEATED_FAILURES,
        ),
        (
            snapshot(queue_depth=79),
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
        ),
        (
            snapshot(queue_depth=80),
            AdminDecision.NOTIFY,
            AdminReasonCode.QUEUE_PRESSURE,
        ),
        (
            snapshot(worker_running=False, failed_jobs=8, queue_depth=100),
            AdminDecision.NOTIFY,
            AdminReasonCode.WORKER_STOPPED,
        ),
        (
            snapshot(failed_jobs=3, queue_depth=100),
            AdminDecision.NOTIFY,
            AdminReasonCode.REPEATED_FAILURES,
        ),
    ],
)
def test_deterministic_policy_priority(
    runtime_snapshot: OperationalSnapshot,
    decision: AdminDecision,
    reason: AdminReasonCode,
) -> None:
    assessment = expected_admin_assessment(runtime_snapshot, AdminPolicy())

    assert assessment.decision is decision
    assert assessment.reason_code is reason
