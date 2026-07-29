from __future__ import annotations

import asyncio
import logging

import pytest

from editorial_team.runtime import (
    QueueCapacityError,
    QueueClosedError,
    QueueNotStartedError,
    RuntimeJobSource,
    RuntimeQueue,
    RuntimeQueueError,
)


def test_start_is_explicit_and_idempotent() -> None:
    async def scenario() -> None:
        queue = RuntimeQueue()
        assert queue._worker is None

        with pytest.raises(QueueNotStartedError):
            await queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="before-start",
                operation=lambda: asyncio.sleep(0),
            )

        await queue.start()
        worker = queue._worker
        assert worker is not None
        await queue.start()
        assert queue._worker is worker
        await queue.close()
        assert queue._worker is None

    asyncio.run(scenario())


def test_fifo_single_worker_and_result_delivery() -> None:
    async def scenario() -> None:
        queue = RuntimeQueue()
        await queue.start()
        order: list[str] = []
        active = 0
        maximum_active = 0

        def operation(name: str, value: int):
            async def run() -> int:
                nonlocal active, maximum_active
                order.append(name)
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0)
                active -= 1
                return value

            return run

        tasks = [
            asyncio.create_task(
                queue.submit(
                    source=RuntimeJobSource.TELEGRAM,
                    correlation_id=f"fifo-{index}",
                    operation=operation(name, index),
                )
            )
            for index, name in enumerate(("first", "second", "third"), start=1)
        ]

        assert await asyncio.gather(*tasks) == [1, 2, 3]
        assert order == ["first", "second", "third"]
        assert maximum_active == 1
        await queue.close()

    asyncio.run(scenario())


def test_job_failure_is_isolated_and_worker_continues() -> None:
    async def scenario() -> None:
        queue = RuntimeQueue()
        await queue.start()
        completed: list[str] = []

        async def succeed(name: str) -> str:
            completed.append(name)
            return name

        async def fail() -> str:
            raise RuntimeError("private failure detail")

        results = await asyncio.gather(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="isolation-1",
                operation=lambda: succeed("first"),
            ),
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="isolation-2",
                operation=fail,
            ),
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="isolation-3",
                operation=lambda: succeed("third"),
            ),
            return_exceptions=True,
        )

        assert results[0] == "first"
        assert isinstance(results[1], RuntimeQueueError)
        assert str(results[1]) == "Runtime job failed"
        assert results[2] == "third"
        assert completed == ["first", "third"]
        assert queue._worker is not None and not queue._worker.done()
        await queue.close()

    asyncio.run(scenario())


def test_bounded_waiting_capacity_rejects_without_discarding_jobs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        queue = RuntimeQueue(capacity=1)
        await queue.start()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        order: list[str] = []

        async def first() -> str:
            order.append("first")
            first_started.set()
            await release_first.wait()
            return "first"

        async def second() -> str:
            order.append("second")
            return "second"

        first_task = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="capacity-1",
                operation=first,
            )
        )
        await first_started.wait()
        second_task = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="capacity-2",
                operation=second,
            )
        )
        while queue._queue.qsize() != 1:
            await asyncio.sleep(0)

        with pytest.raises(QueueCapacityError):
            await queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="capacity-3",
                operation=lambda: asyncio.sleep(0),
            )

        release_first.set()
        assert await asyncio.gather(first_task, second_task) == ["first", "second"]
        assert order == ["first", "second"]
        await queue.close()

    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    asyncio.run(scenario())
    assert "runtime_job_rejected" in caplog.text
    assert "error_category=queue_capacity" in caplog.text


def test_close_drains_accepted_jobs_and_rejects_new_work() -> None:
    async def scenario() -> None:
        queue = RuntimeQueue()
        await queue.start()
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation() -> str:
            started.set()
            await release.wait()
            return "done"

        result_task = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="shutdown-1",
                operation=operation,
            )
        )
        await started.wait()
        close_task = asyncio.create_task(queue.close())
        await asyncio.sleep(0)

        with pytest.raises(QueueClosedError):
            await queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="shutdown-2",
                operation=lambda: asyncio.sleep(0),
            )

        release.set()
        assert await result_task == "done"
        await close_task
        await queue.close()
        assert queue._worker is None
        assert not [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "editorial-runtime-worker" and not task.done()
        ]

    asyncio.run(scenario())


def test_queue_tracing_is_structured_and_excludes_operation_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        queue = RuntimeQueue()
        await queue.start()

        async def succeed() -> str:
            return "safe"

        async def fail() -> None:
            raise RuntimeError("PAYLOAD-AND-EXCEPTION-SECRET")

        result = await asyncio.gather(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="safe-success",
                operation=succeed,
            ),
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="safe-correlation",
                operation=fail,
            ),
            return_exceptions=True,
        )
        assert result[0] == "safe"
        assert isinstance(result[1], RuntimeQueueError)
        assert "PAYLOAD-AND-EXCEPTION-SECRET" not in str(result[1])
        await queue.close()

    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    asyncio.run(scenario())

    trace = caplog.text
    assert "runtime_job_enqueued" in trace
    assert "runtime_job_started" in trace
    assert "runtime_job_completed" in trace
    assert "runtime_job_failed" in trace
    assert "source=telegram" in trace
    assert "capacity=100" in trace
    assert "error_category=runtime_error" in trace
    assert "PAYLOAD-AND-EXCEPTION-SECRET" not in trace
    assert "operation=" not in trace
