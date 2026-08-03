"""Validated, content-free operational heartbeat contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from editorial_team.contracts.common import require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier


class AdminDecision(StrEnum):
    """Whether a future heartbeat should produce a maintainer notification."""

    SILENCE = "silence"
    NOTIFY = "notify"


class AdminReasonCode(StrEnum):
    """Fixed, content-free reasons for an operational decision."""

    SYSTEM_HEALTHY = "system_healthy"
    WORKER_STOPPED = "worker_stopped"
    REPEATED_FAILURES = "repeated_failures"
    QUEUE_PRESSURE = "queue_pressure"


_ALERT_REASONS = frozenset(
    {
        AdminReasonCode.WORKER_STOPPED,
        AdminReasonCode.REPEATED_FAILURES,
        AdminReasonCode.QUEUE_PRESSURE,
    }
)


def validate_admin_decision(
    decision: AdminDecision,
    reason_code: AdminReasonCode,
) -> None:
    """Enforce the shared decision/reason contract."""

    if not isinstance(decision, AdminDecision):
        raise ValueError("decision must be an AdminDecision")
    if not isinstance(reason_code, AdminReasonCode):
        raise ValueError("reason_code must be an AdminReasonCode")
    if decision is AdminDecision.SILENCE and reason_code is not AdminReasonCode.SYSTEM_HEALTHY:
        raise ValueError("SILENCE requires SYSTEM_HEALTHY")
    if decision is AdminDecision.NOTIFY and reason_code not in _ALERT_REASONS:
        raise ValueError("NOTIFY requires an alert reason")


def _require_integer(value: int, field_name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be nonnegative")


@dataclass(frozen=True)
class OperationalSnapshot:
    """Safe runtime facts observed during one heartbeat window."""

    observed_at: datetime
    worker_running: bool
    queue_depth: int
    queue_capacity: int
    completed_jobs: int
    failed_jobs: int
    last_success_at: datetime | None = None

    def __post_init__(self) -> None:
        require_utc_timestamp(self.observed_at, "observed_at")
        if not isinstance(self.worker_running, bool):
            raise ValueError("worker_running must be a boolean")
        _require_integer(self.queue_depth, "queue_depth")
        _require_integer(self.queue_capacity, "queue_capacity", positive=True)
        _require_integer(self.completed_jobs, "completed_jobs")
        _require_integer(self.failed_jobs, "failed_jobs")
        if self.queue_depth > self.queue_capacity:
            raise ValueError("queue_depth must not exceed queue_capacity")
        if self.last_success_at is not None:
            require_utc_timestamp(self.last_success_at, "last_success_at")
            if self.last_success_at > self.observed_at:
                raise ValueError("last_success_at must not be after observed_at")


@dataclass(frozen=True)
class HeartbeatResult:
    """One durable operational decision and its delivery state."""

    id: str
    snapshot: OperationalSnapshot
    decision: AdminDecision
    reason_code: AdminReasonCode
    notification_sent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_identifier(self.id, "id"))
        if not isinstance(self.snapshot, OperationalSnapshot):
            raise ValueError("snapshot must be an OperationalSnapshot")
        validate_admin_decision(self.decision, self.reason_code)
        if not isinstance(self.notification_sent, bool):
            raise ValueError("notification_sent must be a boolean")
        if self.decision is AdminDecision.SILENCE:
            if self.notification_sent:
                raise ValueError("SILENCE cannot be marked as notified")


@dataclass(frozen=True)
class AdminAssessment:
    """The AdminAgent's narrow structured output."""

    decision: AdminDecision
    reason_code: AdminReasonCode

    def __post_init__(self) -> None:
        validate_admin_decision(self.decision, self.reason_code)
