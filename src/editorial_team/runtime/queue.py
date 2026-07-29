"""Bounded FIFO execution queue with one asynchronous worker."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar
from uuid import uuid4

from editorial_team.tracing import error_category, trace_runtime_event

DEFAULT_RUNTIME_QUEUE_CAPACITY = 100
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")
_Result = TypeVar("_Result")
Operation = Callable[[], Awaitable[Any]]


class RuntimeQueueError(RuntimeError):
    """Sanitized runtime queue failure."""


class QueueNotStartedError(RuntimeQueueError):
    """The runtime queue worker has not started."""


class QueueClosedError(RuntimeQueueError):
    """The runtime queue no longer accepts work."""


class QueueCapacityError(RuntimeQueueError):
    """The bounded waiting queue has no available slot."""


class RuntimeJobSource(StrEnum):
    """Safe producer categories for shared runtime work."""

    TELEGRAM = "telegram"
    EXTERNAL = "external"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class _QueuedJob:
    job_id: str
    source: RuntimeJobSource
    correlation_id: str
    enqueued_at: float
    operation: Operation = field(repr=False)
    result: asyncio.Future[Any] = field(repr=False)


_STOP = object()


class RuntimeQueue:
    """Accept bounded opaque jobs and execute them serially in FIFO order."""

    def __init__(self, capacity: int = DEFAULT_RUNTIME_QUEUE_CAPACITY) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._queue: asyncio.Queue[_QueuedJob | object] = asyncio.Queue(
            maxsize=capacity
        )
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._closed = False

    async def start(self) -> None:
        """Start exactly one worker; repeated calls are safely idempotent."""

        if self._closed:
            raise QueueClosedError("Runtime queue is closed")
        if self._worker is not None:
            if not self._worker.done():
                return
            raise RuntimeQueueError("Runtime queue worker is unavailable")
        self._accepting = True
        self._worker = asyncio.create_task(
            self._run_worker(),
            name="editorial-runtime-worker",
        )

    async def submit(
        self,
        *,
        source: RuntimeJobSource,
        correlation_id: str,
        operation: Callable[[], Awaitable[_Result]],
    ) -> _Result:
        """Accept one job immediately and await its own result."""

        self._validate_submission(source, correlation_id, operation)
        loop = asyncio.get_running_loop()
        job = _QueuedJob(
            job_id=f"job-{uuid4().hex}",
            source=source,
            correlation_id=correlation_id,
            enqueued_at=loop.time(),
            operation=operation,
            result=loop.create_future(),
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            trace_runtime_event(
                "runtime_job_rejected",
                correlation_id=correlation_id,
                source=source,
                queue_depth=self._queue.qsize(),
                capacity=self.capacity,
                outcome="rejected",
                error_category="queue_capacity",
            )
            raise QueueCapacityError("Runtime queue is at capacity") from None
        trace_runtime_event(
            "runtime_job_enqueued",
            correlation_id=correlation_id,
            job_id=job.job_id,
            source=source,
            queue_depth=self._queue.qsize(),
            capacity=self.capacity,
        )
        return await job.result

    async def close(self) -> None:
        """Stop accepting work, drain accepted jobs, and terminate the worker."""

        if self._closed:
            return
        self._accepting = False
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        await self._queue.join()
        await self._queue.put(_STOP)
        await worker
        self._worker = None

    def _validate_submission(
        self,
        source: RuntimeJobSource,
        correlation_id: str,
        operation: Operation,
    ) -> None:
        if self._closed or not self._accepting:
            if self._closed:
                raise QueueClosedError("Runtime queue is closed")
            raise QueueNotStartedError("Runtime queue is not started")
        if not isinstance(source, RuntimeJobSource):
            raise RuntimeQueueError("Runtime job source is invalid")
        if (
            not isinstance(correlation_id, str)
            or not correlation_id
            or not _SAFE_IDENTIFIER.fullmatch(correlation_id)
        ):
            raise RuntimeQueueError("Runtime correlation ID is invalid")
        if not callable(operation):
            raise RuntimeQueueError("Runtime operation is invalid")

    async def _run_worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                if not isinstance(item, _QueuedJob):
                    raise RuntimeQueueError("Runtime queue item is invalid")
                await self._execute(item)
            finally:
                self._queue.task_done()

    async def _execute(self, job: _QueuedJob) -> None:
        wait_ms = max(
            0,
            round((asyncio.get_running_loop().time() - job.enqueued_at) * 1000),
        )
        trace_runtime_event(
            "runtime_job_started",
            correlation_id=job.correlation_id,
            job_id=job.job_id,
            source=job.source,
            queue_depth=self._queue.qsize(),
            capacity=self.capacity,
            wait_ms=wait_ms,
        )
        try:
            value = await job.operation()
        except Exception as exc:
            trace_runtime_event(
                "runtime_job_failed",
                correlation_id=job.correlation_id,
                job_id=job.job_id,
                source=job.source,
                queue_depth=self._queue.qsize(),
                capacity=self.capacity,
                outcome="failed",
                error_category=error_category(exc),
            )
            if not job.result.done():
                job.result.set_exception(RuntimeQueueError("Runtime job failed"))
            return
        trace_runtime_event(
            "runtime_job_completed",
            correlation_id=job.correlation_id,
            job_id=job.job_id,
            source=job.source,
            queue_depth=self._queue.qsize(),
            capacity=self.capacity,
            outcome="completed",
        )
        if not job.result.done():
            job.result.set_result(value)
