"""Application service for privileged operational assessment and persistence."""

from __future__ import annotations

from collections.abc import Callable

from editorial_team.agents.protocols import AdminAgent
from editorial_team.contracts.identity import validate_identifier
from editorial_team.errors import ServiceError
from editorial_team.operations.models import HeartbeatResult, OperationalSnapshot
from editorial_team.operations.policy import AdminPolicy, expected_admin_assessment
from editorial_team.operations.store import HeartbeatResultStore
from editorial_team.tracing import error_category, trace_runtime_event


class HeartbeatEvaluationError(ServiceError):
    """A sanitized operational evaluation or persistence failure."""


class AdminPolicyMismatchError(HeartbeatEvaluationError):
    """The AdminAgent assessment conflicts with deterministic policy."""


class HeartbeatEvaluationService:
    """Validate one Admin assessment and persist its heartbeat result."""

    def __init__(
        self,
        *,
        admin_agent: AdminAgent,
        store: HeartbeatResultStore,
        policy: AdminPolicy,
        identifier_generator: Callable[[], str],
    ) -> None:
        self._admin_agent = admin_agent
        self._store = store
        self._policy = policy
        self._identifier_generator = identifier_generator

    def evaluate_and_store(
        self,
        snapshot: OperationalSnapshot,
        *,
        correlation_id: str,
    ) -> HeartbeatResult:
        """Evaluate exactly once, validate policy consistency, and persist."""

        if not isinstance(snapshot, OperationalSnapshot):
            raise ValueError("snapshot must be an OperationalSnapshot")
        correlation_id = validate_identifier(correlation_id, "correlation_id")
        safe_snapshot = {
            "worker_running": snapshot.worker_running,
            "queue_depth": snapshot.queue_depth,
            "queue_capacity": snapshot.queue_capacity,
            "completed_jobs": snapshot.completed_jobs,
            "failed_jobs": snapshot.failed_jobs,
        }
        trace_runtime_event(
            "admin_started",
            correlation_id=correlation_id,
            **safe_snapshot,
        )
        try:
            assessment = self._admin_agent.evaluate(snapshot, self._policy)
            expected = expected_admin_assessment(snapshot, self._policy)
            if assessment != expected:
                raise AdminPolicyMismatchError(
                    "Admin assessment conflicts with operational policy"
                )
            trace_runtime_event(
                "admin_completed",
                correlation_id=correlation_id,
                decision=assessment.decision,
                reason_code=assessment.reason_code,
                outcome="completed",
                **safe_snapshot,
            )
            result = HeartbeatResult(
                id=self._identifier_generator(),
                snapshot=snapshot,
                decision=assessment.decision,
                reason_code=assessment.reason_code,
                notification_sent=False,
            )
            self._store.save(result)
        except Exception as exc:
            trace_runtime_event(
                "admin_failed",
                correlation_id=correlation_id,
                outcome="failed",
                error_category=error_category(exc),
                **safe_snapshot,
            )
            if isinstance(exc, HeartbeatEvaluationError):
                raise
            raise HeartbeatEvaluationError("Heartbeat evaluation failed") from None
        trace_runtime_event(
            "heartbeat_result_saved",
            correlation_id=correlation_id,
            result_id=result.id,
            decision=result.decision,
            reason_code=result.reason_code,
            outcome="saved",
            **safe_snapshot,
        )
        return result
