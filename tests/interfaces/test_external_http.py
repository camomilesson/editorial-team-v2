from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from editorial_team.interfaces.external_http import (
    MAX_BRIEF_LENGTH,
    MAX_IDEMPOTENCY_KEY_LENGTH,
    ExternalBriefHttpAdapter,
)
from editorial_team.runtime import (
    QueueCapacityError,
    RuntimeJobSource,
    RuntimeQueue,
)

TOKEN = "placeholder-token"
BRIEF = "Write a concise launch announcement."


class RecordingService:
    def __init__(self, result: str = "Finished copy") -> None:
        self.result = result
        self.calls: list[str] = []

    def process_brief(self, brief: str) -> object:
        self.calls.append(brief)
        return SimpleNamespace(working_draft=self.result)


class ImmediateQueue:
    def __init__(self) -> None:
        self.submissions: list[dict[str, object]] = []

    async def submit(self, **kwargs: object) -> object:
        self.submissions.append(kwargs)
        operation = kwargs["operation"]
        return await operation()  # type: ignore[operator]


class RejectingQueue(ImmediateQueue):
    async def submit(self, **kwargs: object) -> object:
        self.submissions.append(kwargs)
        raise QueueCapacityError("private queue detail")


def headers(
    *,
    token: str | None = TOKEN,
    key: str | None = "request-1",
    content_type: str = "application/json",
) -> dict[str, str]:
    values = {"Content-Type": content_type}
    if token is not None:
        values["Authorization"] = token if token.startswith("Bearer") else f"Bearer {token}"
    if key is not None:
        values["Idempotency-Key"] = key
    return values


def body(brief: object = BRIEF) -> bytes:
    return json.dumps({"brief": brief}).encode()


def adapter(
    service: RecordingService | None = None,
    queue: ImmediateQueue | None = None,
) -> tuple[ExternalBriefHttpAdapter, RecordingService, ImmediateQueue]:
    service = service or RecordingService()
    queue = queue or ImmediateQueue()
    return (
        ExternalBriefHttpAdapter(
            token=TOKEN,
            service=service,  # type: ignore[arg-type]
            runtime_queue=queue,  # type: ignore[arg-type]
        ),
        service,
        queue,
    )


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic placeholder-token", "Bearer", "Bearer incorrect-token"],
)
def test_authentication_rejections_do_not_submit(authorization: str | None) -> None:
    application, service, queue = adapter()
    request_headers = headers()
    if authorization is None:
        request_headers.pop("Authorization")
    else:
        request_headers["Authorization"] = authorization

    response = asyncio.run(application.handle(headers=request_headers, body=body()))

    assert response.status_code == 401
    assert response.body["error"] == "unauthorized"
    assert service.calls == []
    assert queue.submissions == []
    assert TOKEN not in str(response.body)


def test_valid_request_uses_external_queue_and_returns_only_result() -> None:
    application, service, queue = adapter()

    response = asyncio.run(application.handle(headers=headers(), body=body()))

    assert response.status_code == 200
    assert response.body == {"status": "completed", "result": "Finished copy"}
    assert service.calls == [BRIEF]
    assert len(queue.submissions) == 1
    assert queue.submissions[0]["source"] is RuntimeJobSource.EXTERNAL
    assert str(queue.submissions[0]["correlation_id"]).startswith("ext-")
    assert "correlation" not in str(response.body)


@pytest.mark.parametrize(
    ("request_headers", "request_body", "status"),
    [
        (headers(content_type="text/plain"), body(), 415),
        (headers(key=None), body(), 400),
        (headers(key=" "), body(), 400),
        (headers(key="k" * (MAX_IDEMPOTENCY_KEY_LENGTH + 1)), body(), 400),
        (headers(), b"{", 400),
        (headers(), body(" "), 400),
        (headers(), body(42), 400),
        (headers(), body("x" * (MAX_BRIEF_LENGTH + 1)), 400),
        (headers(), json.dumps({"brief": BRIEF, "route": "writer"}).encode(), 400),
    ],
)
def test_invalid_requests_do_not_submit(
    request_headers: dict[str, str],
    request_body: bytes,
    status: int,
) -> None:
    application, service, queue = adapter()

    response = asyncio.run(
        application.handle(headers=request_headers, body=request_body)
    )

    assert response.status_code == status
    assert service.calls == []
    assert queue.submissions == []


def test_completed_request_is_cached_and_conflict_is_rejected() -> None:
    async def scenario() -> tuple[object, object, object, RecordingService, ImmediateQueue]:
        application, service, queue = adapter()
        first = await application.handle(headers=headers(), body=body())
        repeated = await application.handle(headers=headers(), body=body())
        conflict = await application.handle(
            headers=headers(), body=body("Write something different.")
        )
        return first, repeated, conflict, service, queue

    first, repeated, conflict, service, queue = asyncio.run(scenario())

    assert first == repeated
    assert conflict.status_code == 409
    assert service.calls == [BRIEF]
    assert len(queue.submissions) == 1


def test_concurrent_identical_requests_share_one_execution() -> None:
    @dataclass
    class BlockingService(RecordingService):
        entered: threading.Event
        release: threading.Event

        def __init__(self, entered: threading.Event, release: threading.Event) -> None:
            super().__init__()
            self.entered = entered
            self.release = release

        def process_brief(self, brief: str) -> object:
            self.calls.append(brief)
            self.entered.set()
            self.release.wait(timeout=2)
            return SimpleNamespace(working_draft=self.result)

    async def scenario() -> tuple[list[object], RecordingService, ImmediateQueue]:
        entered = threading.Event()
        release = threading.Event()
        service = BlockingService(entered, release)
        application, _, queue = adapter(service)
        first = asyncio.create_task(application.handle(headers=headers(), body=body()))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(application.handle(headers=headers(), body=body()))
        await asyncio.sleep(0)
        release.set()
        return list(await asyncio.gather(first, second)), service, queue

    responses, service, queue = asyncio.run(scenario())

    assert responses[0] == responses[1]
    assert service.calls == [BRIEF]
    assert len(queue.submissions) == 1


def test_idempotency_is_process_local() -> None:
    async def scenario() -> tuple[RecordingService, RecordingService]:
        first, first_service, _ = adapter()
        second, second_service, _ = adapter()
        await first.handle(headers=headers(), body=body())
        await second.handle(headers=headers(), body=body())
        return first_service, second_service

    first_service, second_service = asyncio.run(scenario())

    assert first_service.calls == second_service.calls == [BRIEF]


def test_queue_rejection_returns_503_and_can_be_retried() -> None:
    queue = RejectingQueue()
    application, service, _ = adapter(queue=queue)

    async def scenario() -> tuple[object, object]:
        first = await application.handle(headers=headers(), body=body())
        second = await application.handle(headers=headers(), body=body())
        return first, second

    first, second = asyncio.run(scenario())

    assert first.status_code == second.status_code == 503
    assert service.calls == []
    assert len(queue.submissions) == 2


def test_accepted_failure_is_cached_and_counted_once() -> None:
    class FailingService(RecordingService):
        def process_brief(self, brief: str) -> object:
            self.calls.append(brief)
            raise RuntimeError("raw provider secret")

    async def scenario() -> tuple[object, object, object, FailingService]:
        service = FailingService()
        queue = RuntimeQueue()
        await queue.start()
        application = ExternalBriefHttpAdapter(
            token=TOKEN,
            service=service,  # type: ignore[arg-type]
            runtime_queue=queue,
        )
        first = await application.handle(headers=headers(), body=body())
        repeated = await application.handle(headers=headers(), body=body())
        stats = queue.stats().for_source(RuntimeJobSource.EXTERNAL)
        await queue.close()
        return first, repeated, stats, service

    first, repeated, stats, service = asyncio.run(scenario())

    assert first == repeated
    assert first.status_code == 500
    assert first.body == {
        "error": "workflow_failed",
        "message": "Editorial workflow failed",
    }
    assert stats.failed_jobs == 1
    assert stats.completed_jobs == 0
    assert service.calls == [BRIEF]
    assert "secret" not in str(first.body)


def test_successful_real_queue_execution_counts_once() -> None:
    async def scenario() -> tuple[object, object]:
        service = RecordingService()
        queue = RuntimeQueue()
        await queue.start()
        application = ExternalBriefHttpAdapter(
            token=TOKEN,
            service=service,  # type: ignore[arg-type]
            runtime_queue=queue,
        )
        await application.handle(headers=headers(), body=body())
        stats = queue.stats().for_source(RuntimeJobSource.EXTERNAL)
        await queue.close()
        return stats, service

    stats, service = asyncio.run(scenario())

    assert stats.completed_jobs == 1
    assert stats.failed_jobs == 0
    assert service.calls == [BRIEF]


def test_shutdown_rejects_without_queue_submission() -> None:
    application, service, queue = adapter()
    application.stop_accepting()

    response = asyncio.run(application.handle(headers=headers(), body=body()))

    assert response.status_code == 503
    assert service.calls == []
    assert queue.submissions == []


def test_traces_exclude_brief_token_result_and_raw_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    application, _, _ = adapter(RecordingService("PRIVATE-EDITORIAL-RESULT"))

    asyncio.run(
        application.handle(
            headers=headers(),
            body=body("PRIVATE-BRIEF-CONTENT"),
        )
    )

    for forbidden in (
        TOKEN,
        "PRIVATE-BRIEF-CONTENT",
        "PRIVATE-EDITORIAL-RESULT",
    ):
        assert forbidden not in caplog.text
