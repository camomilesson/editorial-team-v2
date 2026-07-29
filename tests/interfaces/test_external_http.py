from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace

import pytest

from editorial_team.interfaces.external_http import (
    MAX_BRIEF_LENGTH,
    MAX_IDEMPOTENCY_KEY_LENGTH,
    ExternalBriefHttpAdapter,
    ExternalBriefRequestHandler,
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


def test_retryable_cleanup_survives_all_waiter_cancellation() -> None:
    class BlockingRejectingQueue(ImmediateQueue):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, **kwargs: object) -> object:
            self.submissions.append(kwargs)
            self.entered.set()
            await self.release.wait()
            raise QueueCapacityError("private queue detail")

    async def scenario() -> tuple[int, int]:
        queue = BlockingRejectingQueue()
        application, _, _ = adapter(queue=queue)
        first = asyncio.create_task(application.handle(headers=headers(), body=body()))
        second = asyncio.create_task(application.handle(headers=headers(), body=body()))
        await queue.entered.wait()
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        queue.release.set()
        while application._entries:
            await asyncio.sleep(0)
        retry = await application.handle(headers=headers(), body=body())
        return retry.status_code, len(queue.submissions)

    status, submissions = asyncio.run(scenario())

    assert status == 503
    assert submissions == 2


def test_cancelled_waiter_does_not_remove_successful_cached_result() -> None:
    class BlockingQueue(ImmediateQueue):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def submit(self, **kwargs: object) -> object:
            self.submissions.append(kwargs)
            self.entered.set()
            await self.release.wait()
            operation = kwargs["operation"]
            return await operation()  # type: ignore[operator]

    async def scenario() -> tuple[object, object, int]:
        queue = BlockingQueue()
        application, _, _ = adapter(queue=queue)
        cancelled = asyncio.create_task(
            application.handle(headers=headers(), body=body())
        )
        await queue.entered.wait()
        survivor = asyncio.create_task(
            application.handle(headers=headers(), body=body())
        )
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        queue.release.set()
        completed = await survivor
        cached = await application.handle(headers=headers(), body=body())
        return completed, cached, len(queue.submissions)

    completed, cached, submissions = asyncio.run(scenario())

    assert completed == cached
    assert submissions == 1


def test_caller_mutation_does_not_change_cached_response() -> None:
    async def scenario() -> object:
        application, _, _ = adapter()
        first = await application.handle(headers=headers(), body=body())
        assert isinstance(first.body, dict)
        first.body["result"] = "caller mutation"
        return await application.handle(headers=headers(), body=body())

    cached = asyncio.run(scenario())

    assert cached.body["result"] == "Finished copy"


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


def test_shutdown_rejection_is_traced_safely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    application, _, _ = adapter()
    application.stop_accepting()

    asyncio.run(application.handle(headers=headers(), body=body("PRIVATE-BRIEF")))

    assert "external_shutdown_rejected" in caplog.text
    assert TOKEN not in caplog.text
    assert "PRIVATE-BRIEF" not in caplog.text


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer incorrect-token"],
)
def test_transport_authentication_rejects_before_reading_body(
    authorization: str | None,
) -> None:
    application, _, queue = adapter()
    request = ExternalBriefRequestHandler.__new__(ExternalBriefRequestHandler)
    request.path = "/brief"
    request.headers = {"Content-Length": "PRIVATE-MALFORMED"}
    if authorization is not None:
        request.headers["Authorization"] = authorization
    request.rfile = SimpleNamespace(
        read=lambda _: (_ for _ in ()).throw(AssertionError("body was read"))
    )
    responses: list[object] = []
    request._send = responses.append
    request.server = SimpleNamespace(adapter=application)

    request.do_POST()

    assert responses[0].status_code == 401
    assert queue.submissions == []


def test_transport_valid_authentication_precedes_framing_validation() -> None:
    application, _, _ = adapter()
    request = ExternalBriefRequestHandler.__new__(ExternalBriefRequestHandler)
    request.path = "/brief"
    request.headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Length": "PRIVATE-MALFORMED",
    }
    request.rfile = BytesIO(body())
    responses: list[object] = []
    request._send = responses.append
    request.server = SimpleNamespace(adapter=application)

    request.do_POST()

    assert responses[0].status_code == 400
    assert request.rfile.tell() == 0


def test_transport_rejection_trace_excludes_credentials_and_framing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    application, _, _ = adapter()
    request = ExternalBriefRequestHandler.__new__(ExternalBriefRequestHandler)
    request.path = "/brief"
    request.headers = {
        "Authorization": "Bearer PRIVATE-WRONG-TOKEN",
        "Content-Length": "PRIVATE-FRAMING",
    }
    request.rfile = BytesIO(b"PRIVATE-BODY")
    request._send = lambda response: None
    request.server = SimpleNamespace(adapter=application)

    request.do_POST()

    assert "external_transport_rejected" in caplog.text
    assert "error_category=authentication_rejected" in caplog.text
    for forbidden in ("PRIVATE-WRONG-TOKEN", "PRIVATE-FRAMING", "PRIVATE-BODY"):
        assert forbidden not in caplog.text


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
