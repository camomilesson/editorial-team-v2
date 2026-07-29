"""Provider-neutral maintainer notification boundary and deterministic rendering."""

from __future__ import annotations

from typing import Protocol

from editorial_team.contracts.common import timestamp_to_json
from editorial_team.operations.models import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
)

_REASONS = {
    AdminReasonCode.WORKER_STOPPED: "Runtime worker is not running.",
    AdminReasonCode.REPEATED_FAILURES: "The recent failure threshold was reached.",
    AdminReasonCode.QUEUE_PRESSURE: (
        "The runtime queue reached the configured pressure threshold."
    ),
}


class MaintainerNotifier(Protocol):
    """Deliver one validated operational result to a configured destination."""

    async def notify(self, result: HeartbeatResult) -> None:
        """Deliver one deterministic operational alert."""
        ...


def render_admin_notification(result: HeartbeatResult) -> str:
    """Render deterministic Admin text for one NOTIFY result."""

    if not isinstance(result, HeartbeatResult):
        raise ValueError("result must be a HeartbeatResult")
    if result.decision is not AdminDecision.NOTIFY:
        raise ValueError("Only NOTIFY results can be rendered")
    reason = _REASONS.get(result.reason_code)
    if reason is None:
        raise ValueError("Notification reason is invalid")
    snapshot = result.snapshot
    worker = "running" if snapshot.worker_running else "stopped"
    return (
        "Admin\n\n"
        "Editorial Team requires attention.\n\n"
        f"Reason: {reason}\n\n"
        f"Worker: {worker}\n\n"
        f"Queue: {snapshot.queue_depth}/{snapshot.queue_capacity}\n\n"
        f"Completed jobs in window: {snapshot.completed_jobs}\n\n"
        f"Failed jobs in window: {snapshot.failed_jobs}\n\n"
        f"Observed at: {timestamp_to_json(snapshot.observed_at)}"
    )
