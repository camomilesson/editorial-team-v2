"""Tests for fixed-delay, non-overlapping heartbeat scheduling."""

import asyncio
from datetime import UTC, datetime

import pytest

from editorial_team.operations import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
    HeartbeatScheduler,
    HeartbeatSchedulerError,
    OperationalSnapshot,
)

RESULT = HeartbeatResult(
    "scheduled-result",
    OperationalSnapshot(
        observed_at=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        worker_running=True,
        queue_depth=0,
        queue_capacity=100,
        completed_jobs=0,
        failed_jobs=0,
    ),
    AdminDecision.SILENCE,
    AdminReasonCode.SYSTEM_HEALTHY,
)


class Runner:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0
        self.active = 0
        self.maximum_active = 0

    async def run_once(self, correlation_id: str | None = None) -> HeartbeatResult:
        self.calls += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if self.calls <= self.failures:
            raise RuntimeError("SCHEDULED-SECRET")
        return RESULT


def test_first_run_waits_full_interval_and_failures_do_not_kill_loop() -> None:
    async def scenario() -> None:
        waits: list[float] = []
        decisions = iter([False, False, True])

        async def waiter(stop: asyncio.Event, interval: float) -> bool:
            waits.append(interval)
            await asyncio.sleep(0)
            return next(decisions)

        runner = Runner(failures=1)
        scheduler = HeartbeatScheduler(runner, interval_seconds=0.01, waiter=waiter)
        await scheduler.start()
        assert runner.calls == 0
        task = scheduler._task
        await task
        assert runner.calls == 2
        assert waits == [0.01, 0.01, 0.01]
        await scheduler.close()

    asyncio.run(scenario())


def test_start_is_idempotent_trigger_is_serialized_and_close_is_safe() -> None:
    async def scenario() -> None:
        async def waiter(stop: asyncio.Event, interval: float) -> bool:
            await stop.wait()
            return True

        runner = Runner()
        scheduler = HeartbeatScheduler(runner, interval_seconds=0.01, waiter=waiter)
        await scheduler.start()
        task = scheduler._task
        await scheduler.start()
        assert scheduler._task is task

        results = await asyncio.gather(
            scheduler.trigger_now(),
            scheduler.trigger_now(),
        )
        assert results == [RESULT, RESULT]
        assert runner.maximum_active == 1
        await scheduler.close()
        await scheduler.close()
        assert scheduler._task is None
        assert not [
            item
            for item in asyncio.all_tasks()
            if item.get_name() == "editorial-heartbeat-scheduler" and not item.done()
        ]
        with pytest.raises(HeartbeatSchedulerError):
            await scheduler.trigger_now()

    asyncio.run(scenario())
