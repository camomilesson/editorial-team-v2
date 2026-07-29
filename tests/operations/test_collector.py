"""Tests for safe product-job observation windows."""

from datetime import UTC, datetime, timedelta

from editorial_team.operations import OperationalSnapshotCollector
from editorial_team.runtime import (
    RuntimeJobSource,
    RuntimeQueueStats,
    RuntimeSourceStats,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def source(
    kind: RuntimeJobSource,
    completed: int,
    failed: int,
    last_success_at: datetime | None = None,
) -> RuntimeSourceStats:
    return RuntimeSourceStats(kind, completed, failed, last_success_at)


def stats(
    *,
    telegram: tuple[int, int] = (0, 0),
    external: tuple[int, int] = (0, 0),
    heartbeat: tuple[int, int] = (0, 0),
    worker_running: bool = True,
    depth: int = 2,
) -> RuntimeQueueStats:
    return RuntimeQueueStats(
        worker_running=worker_running,
        waiting_depth=depth,
        capacity=100,
        sources=(
            source(RuntimeJobSource.TELEGRAM, *telegram, NOW - timedelta(minutes=2)),
            source(RuntimeJobSource.EXTERNAL, *external, NOW - timedelta(minutes=1)),
            source(RuntimeJobSource.HEARTBEAT, *heartbeat, NOW),
        ),
    )


class FakeQueue:
    def __init__(self, values: list[RuntimeQueueStats]) -> None:
        self.values = iter(values)

    def stats(self) -> RuntimeQueueStats:
        return next(self.values)


def test_first_and_later_observations_use_product_job_deltas() -> None:
    queue = FakeQueue(
        [
            stats(telegram=(4, 1), external=(2, 3), heartbeat=(20, 10)),
            stats(telegram=(6, 2), external=(5, 3), heartbeat=(21, 11)),
        ]
    )
    collector = OperationalSnapshotCollector(queue, clock=lambda: NOW)  # type: ignore[arg-type]

    first = collector.collect()
    second = collector.collect()

    assert (first.completed_jobs, first.failed_jobs) == (6, 4)
    assert (second.completed_jobs, second.failed_jobs) == (5, 1)
    assert first.last_success_at == NOW - timedelta(minutes=1)
    assert first.worker_running is True
    assert first.queue_depth == 2
    assert first.queue_capacity == 100


def test_heartbeat_source_is_excluded_from_every_window() -> None:
    collector = OperationalSnapshotCollector(
        FakeQueue([stats(heartbeat=(99, 88))]),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    result = collector.collect()

    assert result.completed_jobs == 0
    assert result.failed_jobs == 0


def test_counter_decrease_is_treated_as_safe_reset_without_negative_delta() -> None:
    collector = OperationalSnapshotCollector(
        FakeQueue(
            [
                stats(telegram=(10, 8), external=(5, 4)),
                stats(telegram=(2, 1), external=(1, 0)),
            ]
        ),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    collector.collect()
    reset_window = collector.collect()

    assert reset_window.completed_jobs == 3
    assert reset_window.failed_jobs == 1
    assert reset_window.completed_jobs >= 0
    assert reset_window.failed_jobs >= 0
