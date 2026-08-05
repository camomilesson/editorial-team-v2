"""Tests for optional heartbeat startup and shutdown ordering."""

import asyncio

import pytest

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


class Service:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def close(self) -> None:
        self.order.append("service-close")


def test_enabled_lifecycle_initializes_after_queue_and_stops_before_queue() -> None:
    async def scenario() -> list[str]:
        order: list[str] = []
        adapter = TelegramAdapter(  # type: ignore[arg-type]
            Service(order), Queue(order), allowed_chat_ids=frozenset({123})
        )
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
        "service-close",
    ]


def test_disabled_lifecycle_only_starts_and_closes_queue() -> None:
    async def scenario() -> list[str]:
        order: list[str] = []
        adapter = TelegramAdapter(  # type: ignore[arg-type]
            Service(order), Queue(order), allowed_chat_ids=frozenset({123})
        )
        await adapter.start_runtime(object())  # type: ignore[arg-type]
        await adapter.close_runtime(object())  # type: ignore[arg-type]
        return order

    assert asyncio.run(scenario()) == ["queue-start", "queue-close", "service-close"]


def test_shutdown_failure_still_closes_queue_and_service() -> None:
    class FailingScheduler(Scheduler):
        async def close(self) -> None:
            self.order.append("scheduler-close")
            raise RuntimeError("scheduler failed")

    async def scenario() -> list[str]:
        order: list[str] = []
        adapter = TelegramAdapter(  # type: ignore[arg-type]
            Service(order), Queue(order), allowed_chat_ids=frozenset({123})
        )
        adapter.configure_heartbeat(store=Store(order), scheduler=FailingScheduler(order))
        with pytest.raises(RuntimeError, match="scheduler failed"):
            await adapter.close_runtime(object())  # type: ignore[arg-type]
        return order

    assert asyncio.run(scenario()) == ["scheduler-close", "queue-close", "service-close"]
