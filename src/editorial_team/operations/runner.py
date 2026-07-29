"""Queued heartbeat orchestration across evaluation, persistence, and delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from editorial_team.contracts.identity import validate_identifier
from editorial_team.errors import ServiceError
from editorial_team.operations.collector import OperationalSnapshotCollector
from editorial_team.operations.models import AdminDecision, HeartbeatResult
from editorial_team.operations.notification import MaintainerNotifier
from editorial_team.operations.service import HeartbeatEvaluationService
from editorial_team.operations.store import HeartbeatResultStore
from editorial_team.runtime import RuntimeJobSource, RuntimeQueue
from editorial_team.tracing import error_category, trace_runtime_event


class HeartbeatRunnerError(ServiceError):
    """A heartbeat run failed without exposing transport or provider details."""


class HeartbeatRunner:
    """Run one complete heartbeat as a serialized shared-queue operation."""

    def __init__(
        self,
        *,
        runtime_queue: RuntimeQueue,
        collector: OperationalSnapshotCollector,
        evaluation_service: HeartbeatEvaluationService,
        store: HeartbeatResultStore,
        notifier: MaintainerNotifier,
        correlation_id_generator: Callable[[], str] = (
            lambda: f"heartbeat-{uuid4().hex}"
        ),
    ) -> None:
        self._runtime_queue = runtime_queue
        self._collector = collector
        self._evaluation_service = evaluation_service
        self._store = store
        self._notifier = notifier
        self._correlation_id_generator = correlation_id_generator

    async def run_once(self, correlation_id: str | None = None) -> HeartbeatResult:
        """Collect before enqueueing, then run the full branch exactly once."""

        correlation_id = validate_identifier(
            correlation_id or self._correlation_id_generator(),
            "correlation_id",
        )
        snapshot = self._collector.collect()
        safe_snapshot = {
            "worker_running": snapshot.worker_running,
            "queue_depth": snapshot.queue_depth,
            "queue_capacity": snapshot.queue_capacity,
            "completed_jobs": snapshot.completed_jobs,
            "failed_jobs": snapshot.failed_jobs,
        }
        trace_runtime_event(
            "heartbeat_snapshot_collected",
            correlation_id=correlation_id,
            **safe_snapshot,
        )
        operation_started = False

        async def operation() -> HeartbeatResult:
            nonlocal operation_started
            operation_started = True
            trace_runtime_event(
                "heartbeat_started",
                correlation_id=correlation_id,
                **safe_snapshot,
            )
            try:
                result = await asyncio.to_thread(
                    self._evaluation_service.evaluate_and_store,
                    snapshot,
                    correlation_id=correlation_id,
                )
                if result.decision is AdminDecision.SILENCE:
                    trace_runtime_event(
                        "heartbeat_silenced",
                        correlation_id=correlation_id,
                        result_id=result.id,
                        decision=result.decision,
                        reason_code=result.reason_code,
                        notification_sent=False,
                    )
                    final_result = result
                else:
                    trace_runtime_event(
                        "heartbeat_notification_started",
                        correlation_id=correlation_id,
                        result_id=result.id,
                        reason_code=result.reason_code,
                    )
                    try:
                        await self._notifier.notify(result)
                    except Exception as exc:
                        trace_runtime_event(
                            "heartbeat_notification_failed",
                            correlation_id=correlation_id,
                            result_id=result.id,
                            outcome="failed",
                            error_category=error_category(exc),
                        )
                        raise HeartbeatRunnerError(
                            "Heartbeat notification failed"
                        ) from None
                    trace_runtime_event(
                        "heartbeat_notification_completed",
                        correlation_id=correlation_id,
                        result_id=result.id,
                        outcome="completed",
                    )
                    final_result = await asyncio.to_thread(
                        self._store.mark_notification_sent,
                        result.id,
                    )
            except Exception as exc:
                trace_runtime_event(
                    "heartbeat_failed",
                    correlation_id=correlation_id,
                    outcome="failed",
                    error_category=error_category(exc),
                    **safe_snapshot,
                )
                if isinstance(exc, HeartbeatRunnerError):
                    raise
                raise HeartbeatRunnerError("Heartbeat run failed") from None
            trace_runtime_event(
                "heartbeat_completed",
                correlation_id=correlation_id,
                result_id=final_result.id,
                decision=final_result.decision,
                reason_code=final_result.reason_code,
                notification_sent=final_result.notification_sent,
                outcome="completed",
                **safe_snapshot,
            )
            return final_result

        try:
            return await self._runtime_queue.submit(
                source=RuntimeJobSource.HEARTBEAT,
                correlation_id=correlation_id,
                operation=operation,
            )
        except Exception as exc:
            if not operation_started:
                trace_runtime_event(
                    "heartbeat_failed",
                    correlation_id=correlation_id,
                    outcome="failed",
                    error_category=error_category(exc),
                    **safe_snapshot,
                )
            raise HeartbeatRunnerError("Heartbeat run failed") from None
