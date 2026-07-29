"""Fixed-delay asynchronous scheduling for non-overlapping heartbeats."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Protocol

from editorial_team.errors import ServiceError
from editorial_team.operations.models import HeartbeatResult
from editorial_team.tracing import error_category, trace_runtime_event

StopWaiter = Callable[[asyncio.Event, float], Awaitable[bool]]


class ScheduledHeartbeatRunner(Protocol):
    """The narrow runner operation required by the scheduler."""

    async def run_once(self, correlation_id: str | None = None) -> HeartbeatResult: ...


class HeartbeatSchedulerError(ServiceError):
    """A sanitized scheduler lifecycle or triggered-run failure."""


async def _wait_for_stop(stop: asyncio.Event, interval: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=interval)
    except TimeoutError:
        return False
    return True


class HeartbeatScheduler:
    """Run heartbeats with wait-run-wait fixed-delay semantics."""

    def __init__(
        self,
        runner: ScheduledHeartbeatRunner,
        *,
        interval_seconds: float,
        waiter: StopWaiter = _wait_for_stop,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a positive finite number")
        if not callable(waiter):
            raise ValueError("waiter must be callable")
        self._runner = runner
        self.interval_seconds = float(interval_seconds)
        self._waiter = waiter
        self._stop = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Start exactly one scheduler task without running immediately."""

        if self._closed:
            raise HeartbeatSchedulerError("Heartbeat scheduler is closed")
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run_loop(),
            name="editorial-heartbeat-scheduler",
        )
        trace_runtime_event(
            "heartbeat_scheduler_started",
            correlation_id="heartbeat-scheduler",
            interval_seconds=self.interval_seconds,
        )

    async def trigger_now(self) -> HeartbeatResult:
        """Run once through the same non-overlapping scheduler boundary."""

        if self._closed:
            raise HeartbeatSchedulerError("Heartbeat scheduler is closed")
        async with self._run_lock:
            try:
                return await self._runner.run_once()
            except Exception:
                raise HeartbeatSchedulerError("Triggered heartbeat failed") from None

    async def close(self) -> None:
        """Stop future runs and await any in-flight heartbeat."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        task = self._task
        if task is not None:
            await task
            self._task = None
        trace_runtime_event(
            "heartbeat_scheduler_stopped",
            correlation_id="heartbeat-scheduler",
            outcome="stopped",
        )

    async def _run_loop(self) -> None:
        while not await self._waiter(self._stop, self.interval_seconds):
            trace_runtime_event(
                "heartbeat_scheduled",
                correlation_id="heartbeat-scheduler",
                interval_seconds=self.interval_seconds,
            )
            async with self._run_lock:
                try:
                    await self._runner.run_once()
                except Exception as exc:
                    trace_runtime_event(
                        "heartbeat_scheduler_failed",
                        correlation_id="heartbeat-scheduler",
                        outcome="failed",
                        error_category=error_category(exc),
                    )
