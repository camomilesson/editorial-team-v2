"""Offline tests for the intentional heartbeat alert demo."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from editorial_team.agents import LlmAdminAgent
from editorial_team.models import FakeModelClient, ModelResponse
from editorial_team.operations import (
    AdminPolicy,
    HeartbeatEvaluationService,
    HeartbeatResult,
    HeartbeatRunnerError,
)
from editorial_team.runtime import RuntimeQueue

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "demo_heartbeat_notify.py"
_SPEC = importlib.util.spec_from_file_location("demo_heartbeat_notify", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Manual heartbeat script could not be loaded")
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)
execute_demo_heartbeat = _SCRIPT.execute_demo_heartbeat
format_result = _SCRIPT.format_result
generate_demo_correlation_id = _SCRIPT.generate_demo_correlation_id
require_demo_opt_in = _SCRIPT.require_demo_opt_in
announce_demo_delivery = _SCRIPT.announce_demo_delivery

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class Store:
    def __init__(self) -> None:
        self.saved: list[HeartbeatResult] = []
        self.marked: list[str] = []

    def save(self, result: HeartbeatResult) -> None:
        self.saved.append(result)

    def mark_notification_sent(self, result_id: str) -> HeartbeatResult:
        self.marked.append(result_id)
        original = self.saved[0]
        return HeartbeatResult(
            original.id,
            original.snapshot,
            original.decision,
            original.reason_code,
            notification_sent=True,
        )


class Notifier:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls: list[HeartbeatResult] = []

    async def notify(self, result: HeartbeatResult) -> None:
        self.calls.append(result)
        if self.failure:
            raise RuntimeError("LIVE-DELIVERY-SECRET")


def model() -> FakeModelClient:
    return FakeModelClient(
        [
            ModelResponse(
                json.dumps(
                    {
                        "decision": "notify",
                        "reason_code": "repeated_failures",
                    }
                ),
                (),
                None,
            )
        ]
    )


def service(
    fake_model: FakeModelClient,
    store: Store,
) -> HeartbeatEvaluationService:
    return HeartbeatEvaluationService(
        admin_agent=LlmAdminAgent(fake_model),
        store=store,  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "manual-result",
    )


def test_demo_flow_uses_normal_admin_validation_and_marks_after_one_send() -> None:
    async def scenario() -> tuple[
        HeartbeatResult,
        FakeModelClient,
        Store,
        Notifier,
    ]:
        fake_model = model()
        store = Store()
        notifier = Notifier()
        queue = RuntimeQueue()
        await queue.start()
        try:
            result = await execute_demo_heartbeat(
                runtime_queue=queue,
                evaluation_service=service(fake_model, store),
                store=store,  # type: ignore[arg-type]
                notifier=notifier,
                clock=lambda: NOW,
                correlation_id_generator=lambda: "demo-heartbeat-notify-test",
            )
        finally:
            await queue.close()
        return result, fake_model, store, notifier

    result, fake_model, store, notifier = asyncio.run(scenario())

    assert len(fake_model.requests) == 1
    assert len(store.saved) == 1
    assert notifier.calls == [store.saved[0]]
    assert store.marked == ["manual-result"]
    assert result.notification_sent is True
    assert result.snapshot.observed_at == NOW
    assert result.snapshot.worker_running is True
    assert result.snapshot.queue_depth == 0
    assert result.snapshot.queue_capacity == 100
    assert result.snapshot.completed_jobs == 0
    assert result.snapshot.failed_jobs == 3
    assert format_result(result) == (
        "decision=notify reason_code=repeated_failures notification_sent=true"
    )


def test_delivery_failure_stores_unsent_and_never_marks() -> None:
    async def scenario() -> tuple[Store, Notifier]:
        fake_model = model()
        store = Store()
        notifier = Notifier(failure=True)
        queue = RuntimeQueue()
        await queue.start()
        try:
            with pytest.raises(HeartbeatRunnerError):
                await execute_demo_heartbeat(
                    runtime_queue=queue,
                    evaluation_service=service(fake_model, store),
                    store=store,  # type: ignore[arg-type]
                    notifier=notifier,
                    clock=lambda: NOW,
                    correlation_id_generator=lambda: "demo-heartbeat-notify-test",
                )
        finally:
            await queue.close()
        return store, notifier

    store, notifier = asyncio.run(scenario())

    assert len(store.saved) == 1
    assert store.saved[0].notification_sent is False
    assert len(notifier.calls) == 1
    assert store.marked == []


def test_correlation_ids_are_safe_unique_and_not_part_of_result_output() -> None:
    first = generate_demo_correlation_id()
    second = generate_demo_correlation_id()

    assert first.startswith("demo-heartbeat-notify-")
    assert second.startswith("demo-heartbeat-notify-")
    assert first != second
    assert len(first.removeprefix("demo-heartbeat-notify-")) == 32


@pytest.mark.parametrize("value", [None, "", "false", "1", "yes", "true-value"])
def test_demo_refuses_without_exact_normalized_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("EDITORIAL_HEARTBEAT_DEMO_NOTIFY", raising=False)
    else:
        monkeypatch.setenv("EDITORIAL_HEARTBEAT_DEMO_NOTIFY", value)

    with pytest.raises(_SCRIPT.HeartbeatDemoError, match="explicit opt-in"):
        require_demo_opt_in()


def test_demo_accepts_normalized_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_DEMO_NOTIFY", " TRUE ")

    require_demo_opt_in()


def test_preflight_announcement_is_exact(capsys: pytest.CaptureFixture[str]) -> None:
    announce_demo_delivery()

    assert capsys.readouterr().out == "sending_synthetic_heartbeat_alert=true\n"
