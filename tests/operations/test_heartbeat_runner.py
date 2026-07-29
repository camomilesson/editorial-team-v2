"""Tests for one complete heartbeat through the shared queue boundary."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from editorial_team.operations import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
    HeartbeatRunner,
    HeartbeatRunnerError,
    OperationalSnapshot,
)
from editorial_team.runtime import QueueCapacityError, RuntimeJobSource, RuntimeQueue

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def snapshot(**changes: object) -> OperationalSnapshot:
    values = {
        "observed_at": NOW,
        "worker_running": True,
        "queue_depth": 0,
        "queue_capacity": 100,
        "completed_jobs": 1,
        "failed_jobs": 0,
        "last_success_at": NOW,
    }
    values.update(changes)
    return OperationalSnapshot(**values)  # type: ignore[arg-type]


class Collector:
    def __init__(self, value: OperationalSnapshot, order: list[str]) -> None:
        self.value = value
        self.order = order
        self.calls = 0

    def collect(self) -> OperationalSnapshot:
        self.calls += 1
        self.order.append("collect")
        return self.value


class Queue:
    def __init__(self, order: list[str], *, reject: bool = False) -> None:
        self.order = order
        self.reject = reject
        self.calls: list[dict[str, object]] = []

    async def submit(self, **kwargs: object) -> HeartbeatResult:
        self.order.append("submit")
        self.calls.append(kwargs)
        if self.reject:
            raise QueueCapacityError("queue full")
        operation = kwargs["operation"]
        return await operation()  # type: ignore[operator]


class Store:
    def __init__(self, *, mark_failure: bool = False) -> None:
        self.saved: list[HeartbeatResult] = []
        self.marked: list[str] = []
        self.mark_failure = mark_failure

    def mark_notification_sent(self, result_id: str) -> HeartbeatResult:
        self.marked.append(result_id)
        if self.mark_failure:
            raise RuntimeError("SQL-PATH-SECRET")
        original = self.saved[0]
        return HeartbeatResult(
            original.id,
            original.snapshot,
            original.decision,
            original.reason_code,
            notification_sent=True,
        )


class Evaluation:
    def __init__(
        self,
        store: Store,
        decision: AdminDecision,
        reason: AdminReasonCode,
        *,
        failure: bool = False,
    ) -> None:
        self.store = store
        self.decision = decision
        self.reason = reason
        self.failure = failure
        self.calls: list[tuple[OperationalSnapshot, str]] = []

    def evaluate_and_store(
        self,
        runtime_snapshot: OperationalSnapshot,
        *,
        correlation_id: str,
    ) -> HeartbeatResult:
        self.calls.append((runtime_snapshot, correlation_id))
        if self.failure:
            raise RuntimeError("MODEL-OR-DB-SECRET")
        result = HeartbeatResult(
            "heartbeat-result-1",
            runtime_snapshot,
            self.decision,
            self.reason,
        )
        self.store.saved.append(result)
        return result


class Notifier:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls: list[HeartbeatResult] = []

    async def notify(self, result: HeartbeatResult) -> None:
        self.calls.append(result)
        if self.failure:
            raise RuntimeError("TELEGRAM-CHAT-SECRET")


def build_runner(
    runtime_snapshot: OperationalSnapshot,
    decision: AdminDecision,
    reason: AdminReasonCode,
    *,
    reject: bool = False,
    evaluation_failure: bool = False,
    notification_failure: bool = False,
    mark_failure: bool = False,
) -> tuple[HeartbeatRunner, Collector, Queue, Evaluation, Store, Notifier, list[str]]:
    order: list[str] = []
    collector = Collector(runtime_snapshot, order)
    queue = Queue(order, reject=reject)
    store = Store(mark_failure=mark_failure)
    evaluation = Evaluation(
        store,
        decision,
        reason,
        failure=evaluation_failure,
    )
    notifier = Notifier(failure=notification_failure)
    runner = HeartbeatRunner(
        runtime_queue=queue,  # type: ignore[arg-type]
        collector=collector,  # type: ignore[arg-type]
        evaluation_service=evaluation,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        notifier=notifier,
        correlation_id_generator=lambda: "heartbeat-generated",
    )
    return runner, collector, queue, evaluation, store, notifier, order


def test_silence_is_queued_persisted_and_never_notified() -> None:
    runner, collector, queue, evaluation, store, notifier, order = build_runner(
        snapshot(),
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
    )

    result = asyncio.run(runner.run_once())

    assert order == ["collect", "submit"]
    assert collector.calls == 1
    assert queue.calls[0]["source"] is RuntimeJobSource.HEARTBEAT
    assert queue.calls[0]["correlation_id"] == "heartbeat-generated"
    assert len(evaluation.calls) == len(store.saved) == 1
    assert notifier.calls == []
    assert store.marked == []
    assert result.notification_sent is False


@pytest.mark.parametrize(
    ("runtime_snapshot", "reason"),
    [
        (snapshot(worker_running=False), AdminReasonCode.WORKER_STOPPED),
        (snapshot(failed_jobs=3), AdminReasonCode.REPEATED_FAILURES),
        (snapshot(queue_depth=80), AdminReasonCode.QUEUE_PRESSURE),
    ],
)
def test_notify_sends_once_then_marks_sent(
    runtime_snapshot: OperationalSnapshot,
    reason: AdminReasonCode,
) -> None:
    runner, _, _, _, store, notifier, _ = build_runner(
        runtime_snapshot,
        AdminDecision.NOTIFY,
        reason,
    )

    result = asyncio.run(runner.run_once("heartbeat-explicit"))

    assert len(store.saved) == 1
    assert notifier.calls == [store.saved[0]]
    assert store.marked == ["heartbeat-result-1"]
    assert result.notification_sent is True


def test_queue_rejection_does_not_evaluate_persist_or_notify() -> None:
    runner, _, _, evaluation, store, notifier, _ = build_runner(
        snapshot(),
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
        reject=True,
    )

    with pytest.raises(HeartbeatRunnerError):
        asyncio.run(runner.run_once())

    assert evaluation.calls == []
    assert store.saved == []
    assert notifier.calls == []


def test_evaluation_failure_does_not_notify() -> None:
    runner, _, _, evaluation, store, notifier, _ = build_runner(
        snapshot(),
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
        evaluation_failure=True,
    )

    with pytest.raises(HeartbeatRunnerError) as error:
        asyncio.run(runner.run_once())

    assert str(error.value) == "Heartbeat run failed"
    assert len(evaluation.calls) == 1
    assert store.saved == []
    assert notifier.calls == []


def test_delivery_failure_leaves_durable_unsent_result() -> None:
    runner, _, _, _, store, notifier, _ = build_runner(
        snapshot(failed_jobs=3),
        AdminDecision.NOTIFY,
        AdminReasonCode.REPEATED_FAILURES,
        notification_failure=True,
    )

    with pytest.raises(HeartbeatRunnerError):
        asyncio.run(runner.run_once())

    assert len(store.saved) == 1
    assert store.saved[0].notification_sent is False
    assert len(notifier.calls) == 1
    assert store.marked == []


def test_mark_failure_sends_once_and_surfaces_without_duplicate() -> None:
    runner, _, _, _, store, notifier, _ = build_runner(
        snapshot(queue_depth=80),
        AdminDecision.NOTIFY,
        AdminReasonCode.QUEUE_PRESSURE,
        mark_failure=True,
    )

    with pytest.raises(HeartbeatRunnerError):
        asyncio.run(runner.run_once())

    assert len(notifier.calls) == 1
    assert store.marked == ["heartbeat-result-1"]
    assert store.saved[0].notification_sent is False


def test_heartbeat_does_not_interleave_with_existing_product_job() -> None:
    async def scenario() -> list[str]:
        order: list[str] = []
        queue = RuntimeQueue()
        await queue.start()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def product_job() -> None:
            order.append("product-start")
            entered.set()
            await release.wait()
            order.append("product-end")

        product = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="product-job",
                operation=product_job,
            )
        )
        await entered.wait()
        collector = Collector(snapshot(), order)
        store = Store()

        class OrderedEvaluation(Evaluation):
            def evaluate_and_store(
                self,
                runtime_snapshot: OperationalSnapshot,
                *,
                correlation_id: str,
            ) -> HeartbeatResult:
                order.append("heartbeat-operation")
                return super().evaluate_and_store(
                    runtime_snapshot,
                    correlation_id=correlation_id,
                )

        runner = HeartbeatRunner(
            runtime_queue=queue,
            collector=collector,  # type: ignore[arg-type]
            evaluation_service=OrderedEvaluation(
                store,
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            ),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            notifier=Notifier(),
        )
        heartbeat = asyncio.create_task(runner.run_once("serialized-heartbeat"))
        while queue.stats().waiting_depth != 1:
            await asyncio.sleep(0)
        assert "heartbeat-operation" not in order
        release.set()
        await asyncio.gather(product, heartbeat)
        await queue.close()
        return order

    order = asyncio.run(scenario())

    assert order.index("product-end") < order.index("heartbeat-operation")
    assert order.index("collect") < order.index("heartbeat-operation")


def test_runner_tracing_is_truthful_and_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    runner, _, _, _, _, _, _ = build_runner(
        snapshot(failed_jobs=3),
        AdminDecision.NOTIFY,
        AdminReasonCode.REPEATED_FAILURES,
    )

    asyncio.run(runner.run_once("traced-heartbeat"))

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "editorial_team.live_trace"
    ]
    assert [message.split()[0] for message in messages] == [
        "heartbeat_snapshot_collected",
        "heartbeat_started",
        "heartbeat_notification_started",
        "heartbeat_notification_completed",
        "heartbeat_completed",
    ]
    trace = "\n".join(messages)
    assert "notification_sent=true" in trace
    for secret in (
        "MAINTAINER-CHAT-ID-SECRET",
        "DATABASE-PATH-SECRET",
        "USER-TEXT-SECRET",
        "DRAFT-SECRET",
        "PROMPT-SECRET",
        "MODEL-OUTPUT-SECRET",
    ):
        assert secret not in trace


def test_failed_delivery_traces_no_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    runner, _, _, _, _, _, _ = build_runner(
        snapshot(queue_depth=80),
        AdminDecision.NOTIFY,
        AdminReasonCode.QUEUE_PRESSURE,
        notification_failure=True,
    )

    with pytest.raises(HeartbeatRunnerError):
        asyncio.run(runner.run_once("failed-traced-heartbeat"))

    trace = caplog.text
    assert "heartbeat_notification_failed" in trace
    assert "heartbeat_failed" in trace
    assert "heartbeat_notification_completed" not in trace
    assert "heartbeat_completed" not in trace
    assert "TELEGRAM-CHAT-SECRET" not in trace
