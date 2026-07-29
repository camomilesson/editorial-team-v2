"""SQLite integration for SILENCE, NOTIFY, and failed delivery."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from editorial_team.operations import (
    AdminAssessment,
    AdminDecision,
    AdminPolicy,
    AdminReasonCode,
    HeartbeatEvaluationService,
    HeartbeatRunner,
    HeartbeatRunnerError,
    OperationalSnapshot,
    SQLiteHeartbeatResultStore,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class Queue:
    async def submit(self, **kwargs: object):
        operation = kwargs["operation"]
        return await operation()  # type: ignore[operator]


class Collector:
    def __init__(self, values: list[OperationalSnapshot]) -> None:
        self.values = iter(values)

    def collect(self) -> OperationalSnapshot:
        return next(self.values)


class Admin:
    def evaluate(
        self,
        snapshot: OperationalSnapshot,
        policy: AdminPolicy,
    ) -> AdminAssessment:
        if not snapshot.worker_running:
            return AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.WORKER_STOPPED,
            )
        return AdminAssessment(
            AdminDecision.SILENCE,
            AdminReasonCode.SYSTEM_HEALTHY,
        )


class Notifier:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls = 0

    async def notify(self, result) -> None:
        self.calls += 1
        if self.failure:
            raise RuntimeError("DELIVERY-SECRET")


def operational_snapshot(*, worker_running: bool) -> OperationalSnapshot:
    return OperationalSnapshot(
        observed_at=NOW,
        worker_running=worker_running,
        queue_depth=0,
        queue_capacity=100,
        completed_jobs=2,
        failed_jobs=0,
    )


def service(
    store: SQLiteHeartbeatResultStore,
    ids: Iterator[str],
) -> HeartbeatEvaluationService:
    return HeartbeatEvaluationService(
        admin_agent=Admin(),
        store=store,
        policy=AdminPolicy(),
        identifier_generator=lambda: next(ids),
    )


def test_silence_and_notify_survive_reopen_with_correct_sent_state(
    tmp_path: Path,
) -> None:
    database = tmp_path / "heartbeats.db"
    store = SQLiteHeartbeatResultStore(database)
    store.initialize()
    ids = iter(["silence-result", "notify-result"])
    notifier = Notifier()
    runner = HeartbeatRunner(
        runtime_queue=Queue(),  # type: ignore[arg-type]
        collector=Collector(
            [
                operational_snapshot(worker_running=True),
                operational_snapshot(worker_running=False),
            ]
        ),  # type: ignore[arg-type]
        evaluation_service=service(store, ids),
        store=store,
        notifier=notifier,
    )

    silence = asyncio.run(runner.run_once("integration-silence"))
    notify = asyncio.run(runner.run_once("integration-notify"))

    reopened = SQLiteHeartbeatResultStore(database)
    assert silence.notification_sent is False
    assert notify.notification_sent is True
    assert reopened.get("silence-result").notification_sent is False
    assert reopened.get("notify-result").notification_sent is True
    assert notifier.calls == 1
    with sqlite3.connect(database) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert tables == [("heartbeat_results",)]


def test_failed_delivery_remains_durable_and_unsent(tmp_path: Path) -> None:
    database = tmp_path / "heartbeats.db"
    store = SQLiteHeartbeatResultStore(database)
    store.initialize()
    notifier = Notifier(failure=True)
    runner = HeartbeatRunner(
        runtime_queue=Queue(),  # type: ignore[arg-type]
        collector=Collector([operational_snapshot(worker_running=False)]),  # type: ignore[arg-type]
        evaluation_service=service(store, iter(["failed-notify"])),
        store=store,
        notifier=notifier,
    )

    with pytest.raises(HeartbeatRunnerError):
        asyncio.run(runner.run_once("integration-failed"))

    assert store.get("failed-notify").notification_sent is False
    assert notifier.calls == 1
