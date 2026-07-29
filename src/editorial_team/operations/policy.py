"""Deterministic operational policy for privileged Admin assessments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from editorial_team.operations.models import (
    AdminAssessment,
    AdminDecision,
    AdminReasonCode,
    OperationalSnapshot,
)


@dataclass(frozen=True)
class AdminPolicy:
    """Immutable thresholds supplied to the AdminAgent and application validator."""

    failure_threshold: int = 3
    queue_pressure_ratio: float = 0.8

    def __post_init__(self) -> None:
        if (
            isinstance(self.failure_threshold, bool)
            or not isinstance(self.failure_threshold, int)
            or self.failure_threshold <= 0
        ):
            raise ValueError("failure_threshold must be a positive integer")
        if (
            isinstance(self.queue_pressure_ratio, bool)
            or not isinstance(self.queue_pressure_ratio, (int, float))
            or not math.isfinite(self.queue_pressure_ratio)
            or not 0 < self.queue_pressure_ratio <= 1
        ):
            raise ValueError("queue_pressure_ratio must be greater than 0 and at most 1")
        object.__setattr__(
            self,
            "queue_pressure_ratio",
            float(self.queue_pressure_ratio),
        )


def expected_admin_assessment(
    snapshot: OperationalSnapshot,
    policy: AdminPolicy,
) -> AdminAssessment:
    """Compute the required assessment using the contractual priority order."""

    if not isinstance(snapshot, OperationalSnapshot):
        raise ValueError("snapshot must be an OperationalSnapshot")
    if not isinstance(policy, AdminPolicy):
        raise ValueError("policy must be an AdminPolicy")
    if not snapshot.worker_running:
        return AdminAssessment(
            AdminDecision.NOTIFY,
            AdminReasonCode.WORKER_STOPPED,
        )
    if snapshot.failed_jobs >= policy.failure_threshold:
        return AdminAssessment(
            AdminDecision.NOTIFY,
            AdminReasonCode.REPEATED_FAILURES,
        )
    pressure_ratio = Fraction(str(policy.queue_pressure_ratio))
    if (
        snapshot.queue_depth * pressure_ratio.denominator
        >= snapshot.queue_capacity * pressure_ratio.numerator
    ):
        return AdminAssessment(
            AdminDecision.NOTIFY,
            AdminReasonCode.QUEUE_PRESSURE,
        )
    return AdminAssessment(
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
    )
