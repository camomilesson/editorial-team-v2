"""In-memory observation-window collection from safe runtime queue metrics."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from editorial_team.operations.models import OperationalSnapshot
from editorial_team.runtime import RuntimeJobSource, RuntimeQueue

_PRODUCT_SOURCES = (RuntimeJobSource.TELEGRAM,)


class OperationalSnapshotCollector:
    """Convert cumulative product-job metrics into heartbeat-window deltas."""

    def __init__(
        self,
        runtime_queue: RuntimeQueue,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._runtime_queue = runtime_queue
        self._clock = clock
        self._completed_baseline = {source: 0 for source in _PRODUCT_SOURCES}
        self._failed_baseline = {source: 0 for source in _PRODUCT_SOURCES}

    def collect(self) -> OperationalSnapshot:
        """Return product-job deltas and advance the in-memory baseline.

        If a cumulative counter decreases, it is treated as a process-local
        counter reset and the current value becomes the new window count.
        """

        stats = self._runtime_queue.stats()
        completed = 0
        failed = 0
        last_successes: list[datetime] = []
        for source in _PRODUCT_SOURCES:
            current = stats.for_source(source)
            completed += self._delta(
                current.completed_jobs,
                self._completed_baseline[source],
            )
            failed += self._delta(
                current.failed_jobs,
                self._failed_baseline[source],
            )
            self._completed_baseline[source] = current.completed_jobs
            self._failed_baseline[source] = current.failed_jobs
            if current.last_success_at is not None:
                last_successes.append(current.last_success_at)
        return OperationalSnapshot(
            observed_at=self._clock(),
            worker_running=stats.worker_running,
            queue_depth=stats.waiting_depth,
            queue_capacity=stats.capacity,
            completed_jobs=completed,
            failed_jobs=failed,
            last_success_at=max(last_successes, default=None),
        )

    @staticmethod
    def _delta(current: int, baseline: int) -> int:
        return current - baseline if current >= baseline else current
