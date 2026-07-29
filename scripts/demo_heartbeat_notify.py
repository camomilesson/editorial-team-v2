#!/usr/bin/env python3
"""Intentionally demo one real repeated-failures heartbeat notification."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from editorial_team.app import (
    HeartbeatConfigurationError,
    LiveConfigurationError,
    build_live_application_from_env,
    load_heartbeat_configuration,
)
from editorial_team.errors import ServiceError
from editorial_team.operations import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatEvaluationService,
    HeartbeatResult,
    HeartbeatResultStore,
    HeartbeatRunner,
    MaintainerNotifier,
    OperationalSnapshotCollector,
)
from editorial_team.runtime import (
    RuntimeJobSource,
    RuntimeQueue,
    RuntimeQueueStats,
    RuntimeSourceStats,
)

DEMO_NOTIFY_ENVIRONMENT_VARIABLE = "EDITORIAL_HEARTBEAT_DEMO_NOTIFY"


class HeartbeatDemoError(ServiceError):
    """A sanitized heartbeat demo failure."""


class _SyntheticMetricsSource:
    """Expose one explicit content-free repeated-failures observation."""

    def stats(self) -> RuntimeQueueStats:
        return RuntimeQueueStats(
            worker_running=True,
            waiting_depth=0,
            capacity=100,
            sources=(
                RuntimeSourceStats(RuntimeJobSource.TELEGRAM, 0, 3, None),
                RuntimeSourceStats(RuntimeJobSource.EXTERNAL, 0, 0, None),
                RuntimeSourceStats(RuntimeJobSource.HEARTBEAT, 0, 0, None),
            ),
        )


def generate_demo_correlation_id() -> str:
    """Generate one opaque correlation ID that is never printed."""

    return f"demo-heartbeat-notify-{uuid4().hex}"


def require_demo_opt_in() -> None:
    """Require an explicit normalized true value for real alert delivery."""

    if os.getenv(DEMO_NOTIFY_ENVIRONMENT_VARIABLE, "").strip().lower() != "true":
        raise HeartbeatDemoError("Heartbeat demo requires explicit opt-in")


def announce_demo_delivery() -> None:
    """Print the explicit preflight line immediately before live calls."""

    print("sending_synthetic_heartbeat_alert=true")


async def execute_demo_heartbeat(
    *,
    runtime_queue: RuntimeQueue,
    evaluation_service: HeartbeatEvaluationService,
    store: HeartbeatResultStore,
    notifier: MaintainerNotifier,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    correlation_id_generator: Callable[[], str] = generate_demo_correlation_id,
) -> HeartbeatResult:
    """Run the production heartbeat flow against one explicit safe snapshot."""

    collector = OperationalSnapshotCollector(
        _SyntheticMetricsSource(),  # type: ignore[arg-type]
        clock=clock,
    )
    runner = HeartbeatRunner(
        runtime_queue=runtime_queue,
        collector=collector,
        evaluation_service=evaluation_service,
        store=store,
        notifier=notifier,
    )
    result = await runner.run_once(correlation_id_generator())
    if (
        result.decision is not AdminDecision.NOTIFY
        or result.reason_code is not AdminReasonCode.REPEATED_FAILURES
        or not result.notification_sent
    ):
        raise HeartbeatDemoError("Heartbeat demo returned an unexpected result")
    return result


async def run_from_environment() -> HeartbeatResult:
    """Compose normal live dependencies without starting polling or scheduling."""

    require_demo_opt_in()
    try:
        configuration = load_heartbeat_configuration()
    except HeartbeatConfigurationError:
        raise HeartbeatDemoError("Heartbeat demo configuration is invalid") from None
    if not configuration.enabled:
        raise HeartbeatDemoError("Heartbeat demo requires enabled heartbeat")

    try:
        live = build_live_application_from_env()
    except LiveConfigurationError:
        raise HeartbeatDemoError("Heartbeat demo composition failed") from None
    if live.heartbeat is None:
        raise HeartbeatDemoError("Heartbeat demo composition failed")

    await live.runtime_queue.start()
    try:
        await asyncio.to_thread(live.heartbeat.store.initialize)
        announce_demo_delivery()
        return await execute_demo_heartbeat(
            runtime_queue=live.runtime_queue,
            evaluation_service=live.heartbeat.evaluation_service,
            store=live.heartbeat.store,
            notifier=live.heartbeat.notifier,
        )
    finally:
        await live.runtime_queue.close()


def format_result(result: HeartbeatResult) -> str:
    """Return the only successful user-facing verification output."""

    return (
        f"decision={result.decision.value} "
        f"reason_code={result.reason_code.value} "
        f"notification_sent={str(result.notification_sent).lower()}"
    )


def main() -> None:
    """Run one manual heartbeat without polling or scheduler startup."""

    try:
        result = asyncio.run(run_from_environment())
    except Exception:
        raise SystemExit("Heartbeat alert demo failed") from None
    print(format_result(result))


if __name__ == "__main__":
    main()
