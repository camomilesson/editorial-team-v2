"""Tests for deterministic Telegram maintainer alerts."""

import asyncio
from datetime import UTC, datetime

import pytest

from editorial_team.interfaces.admin import (
    MaintainerNotificationError,
    TelegramMaintainerNotifier,
)
from editorial_team.operations import (
    AdminDecision,
    AdminReasonCode,
    HeartbeatResult,
    OperationalSnapshot,
    render_admin_notification,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def result(reason: AdminReasonCode) -> HeartbeatResult:
    return HeartbeatResult(
        "private-internal-result-id",
        OperationalSnapshot(
            observed_at=NOW,
            worker_running=reason is not AdminReasonCode.WORKER_STOPPED,
            queue_depth=80,
            queue_capacity=100,
            completed_jobs=7,
            failed_jobs=3,
        ),
        AdminDecision.NOTIFY,
        reason,
    )


@pytest.mark.parametrize(
    ("reason", "reason_text", "worker"),
    [
        (
            AdminReasonCode.WORKER_STOPPED,
            "Runtime worker is not running.",
            "stopped",
        ),
        (
            AdminReasonCode.REPEATED_FAILURES,
            "The recent failure threshold was reached.",
            "running",
        ),
        (
            AdminReasonCode.QUEUE_PRESSURE,
            "The runtime queue reached the configured pressure threshold.",
            "running",
        ),
    ],
)
def test_notification_text_is_exact_and_content_free(
    reason: AdminReasonCode,
    reason_text: str,
    worker: str,
) -> None:
    text = render_admin_notification(result(reason))

    assert text == (
        "Admin\n\n"
        "Editorial Team requires attention.\n\n"
        f"Reason: {reason_text}\n\n"
        f"Worker: {worker}\n\n"
        "Queue: 80/100\n\n"
        "Completed jobs in window: 7\n\n"
        "Failed jobs in window: 3\n\n"
        "Observed at: 2026-07-29T12:00:00Z"
    )
    assert "private-internal-result-id" not in text


def test_healthy_result_cannot_be_rendered() -> None:
    healthy = HeartbeatResult(
        "healthy",
        result(AdminReasonCode.QUEUE_PRESSURE).snapshot,
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
    )

    with pytest.raises(ValueError, match="NOTIFY"):
        render_admin_notification(healthy)


class Bot:
    def __init__(self, failure: bool = False) -> None:
        self.failure = failure
        self.sent: list[dict[str, object]] = []

    async def send_message(self, **kwargs: object) -> None:
        if self.failure:
            raise RuntimeError("CHAT-ID-AND-DELIVERY-SECRET")
        self.sent.append(kwargs)


def test_notifier_sends_once_to_injected_destination() -> None:
    bot = Bot()
    notification = result(AdminReasonCode.REPEATED_FAILURES)

    asyncio.run(TelegramMaintainerNotifier(bot, -100123).notify(notification))

    assert bot.sent == [
        {
            "chat_id": -100123,
            "text": render_admin_notification(notification),
        }
    ]


def test_notifier_failure_is_sanitized() -> None:
    with pytest.raises(MaintainerNotificationError) as error:
        asyncio.run(
            TelegramMaintainerNotifier(Bot(failure=True), -100123).notify(
                result(AdminReasonCode.QUEUE_PRESSURE)
            )
        )

    assert str(error.value) == "Maintainer notification delivery failed"
