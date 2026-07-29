"""Tests for optional heartbeat startup and shutdown ordering."""

import asyncio

from editorial_team.interfaces.telegram import TelegramAdapter


class Queue:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def start(self) -> None:
        self.order.append("queue-start")

    async def close(self) -> None:
        self.order.append("queue-close")


class Store:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def initialize(self) -> None:
        self.order.append("store-initialize")


class Scheduler:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def start(self) -> None:
        self.order.append("scheduler-start")

    async def close(self) -> None:
        self.order.append("scheduler-close")


def test_enabled_lifecycle_initializes_after_queue_and_stops_before_queue() -> None:
    async def scenario() -> list[str]:
        order: list[str] = []
        adapter = TelegramAdapter(object(), Queue(order))  # type: ignore[arg-type]
        adapter.configure_heartbeat(
            store=Store(order),
            scheduler=Scheduler(order),
        )

        await adapter.start_runtime(object())  # type: ignore[arg-type]
        await adapter.close_runtime(object())  # type: ignore[arg-type]
        return order

    assert asyncio.run(scenario()) == [
        "queue-start",
        "store-initialize",
        "scheduler-start",
        "scheduler-close",
        "queue-close",
    ]


def test_disabled_lifecycle_only_starts_and_closes_queue() -> None:
    async def scenario() -> list[str]:
        order: list[str] = []
        adapter = TelegramAdapter(object(), Queue(order))  # type: ignore[arg-type]
        await adapter.start_runtime(object())  # type: ignore[arg-type]
        await adapter.close_runtime(object())  # type: ignore[arg-type]
        return order

    assert asyncio.run(scenario()) == ["queue-start", "queue-close"]
