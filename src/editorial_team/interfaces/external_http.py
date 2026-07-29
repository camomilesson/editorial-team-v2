"""Authenticated HTTP adapter for standalone editorial briefs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

from editorial_team.conversation import ConversationService
from editorial_team.runtime import (
    QueueCapacityError,
    QueueClosedError,
    QueueNotStartedError,
    RuntimeJobSource,
    RuntimeQueue,
    RuntimeQueueError,
)
from editorial_team.tracing import (
    TurnTrace,
    bind_turn_trace,
    error_category,
    trace_event,
    trace_runtime_event,
)

MAX_BRIEF_LENGTH = 20_000
MAX_IDEMPOTENCY_KEY_LENGTH = 200
MAX_REQUEST_BODY_BYTES = MAX_BRIEF_LENGTH * 4 + 1_024


@dataclass(frozen=True)
class HttpResponse:
    """Transport-ready JSON response."""

    status_code: int
    body: dict[str, str]


@dataclass
class _IdempotencyEntry:
    fingerprint: str
    execution: asyncio.Task[HttpResponse]


def _error(status_code: int, code: str, message: str) -> HttpResponse:
    return HttpResponse(status_code, {"error": code, "message": message})


class ExternalBriefHttpAdapter:
    """Validate HTTP input and submit accepted briefs to the shared runtime."""

    def __init__(
        self,
        *,
        token: str,
        service: ConversationService,
        runtime_queue: RuntimeQueue,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("External API token is required")
        self._token = token
        self._service = service
        self._runtime_queue = runtime_queue
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = asyncio.Lock()
        self._accepting = True

    def stop_accepting(self) -> None:
        """Reject new HTTP work during shutdown."""

        self._accepting = False

    async def handle(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
    ) -> HttpResponse:
        """Handle one POST /brief request without exposing sensitive input."""

        correlation_id = f"ext-{uuid4().hex[:12]}"
        trace_runtime_event("external_request_received", correlation_id=correlation_id)
        if not self._authorized(headers.get("Authorization")):
            trace_runtime_event(
                "external_authentication_rejected",
                correlation_id=correlation_id,
                outcome="rejected",
            )
            return _error(401, "unauthorized", "Authentication is required")
        if not self._accepting:
            return _error(503, "service_unavailable", "Service is unavailable")

        content_type = headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return self._validation_error(
                correlation_id,
                415,
                "unsupported_content_type",
                "Content-Type must be application/json",
            )
        key = headers.get("Idempotency-Key", "").strip()
        if not key or len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
            return self._validation_error(
                correlation_id,
                400,
                "invalid_idempotency_key",
                "Idempotency-Key is invalid",
            )
        if len(body) > MAX_REQUEST_BODY_BYTES:
            return self._validation_error(
                correlation_id, 400, "invalid_request", "Request body is invalid"
            )
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._validation_error(
                correlation_id, 400, "malformed_json", "Request body is not valid JSON"
            )
        if not isinstance(payload, dict) or set(payload) != {"brief"}:
            return self._validation_error(
                correlation_id, 400, "invalid_request", "Request body is invalid"
            )
        brief = payload["brief"]
        if (
            not isinstance(brief, str)
            or not brief.strip()
            or len(brief) > MAX_BRIEF_LENGTH
        ):
            return self._validation_error(
                correlation_id, 400, "invalid_brief", "Brief is invalid"
            )

        fingerprint = hashlib.sha256(
            json.dumps({"brief": brief}, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if not hmac.compare_digest(entry.fingerprint, fingerprint):
                    trace_runtime_event(
                        "external_idempotency_conflict",
                        correlation_id=correlation_id,
                        outcome="rejected",
                    )
                    return _error(
                        409,
                        "idempotency_conflict",
                        "Idempotency key was used for a different request",
                    )
                trace_runtime_event(
                    "external_idempotency_cache_hit",
                    correlation_id=correlation_id,
                )
                execution = entry.execution
            else:
                execution = asyncio.create_task(
                    self._submit(brief=brief, correlation_id=correlation_id)
                )
                entry = _IdempotencyEntry(fingerprint, execution)
                self._entries[key] = entry

        response = await asyncio.shield(execution)
        if response.status_code == 503:
            async with self._lock:
                if self._entries.get(key) is entry:
                    self._entries.pop(key)
        return response

    def _authorized(self, authorization: str | None) -> bool:
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return False
        supplied = authorization[7:]
        return bool(supplied) and hmac.compare_digest(supplied, self._token)

    def _validation_error(
        self,
        correlation_id: str,
        status_code: int,
        code: str,
        message: str,
    ) -> HttpResponse:
        trace_runtime_event(
            "external_request_validation_rejected",
            correlation_id=correlation_id,
            outcome="rejected",
            error_category=code,
        )
        return _error(status_code, code, message)

    async def _submit(self, *, brief: str, correlation_id: str) -> HttpResponse:
        trace = TurnTrace(correlation_id=correlation_id, update_id=None, stage="external")

        async def operation() -> str:
            with bind_turn_trace(trace):
                trace_event("external_job_submitted", stage="external")
                result = await asyncio.to_thread(self._service.process_brief, brief)
                return result.working_draft

        try:
            result = await self._runtime_queue.submit(
                source=RuntimeJobSource.EXTERNAL,
                correlation_id=correlation_id,
                operation=operation,
            )
        except (QueueCapacityError, QueueClosedError, QueueNotStartedError):
            trace_runtime_event(
                "external_request_failed",
                correlation_id=correlation_id,
                outcome="rejected",
                error_category="service_unavailable",
            )
            return _error(503, "service_unavailable", "Service is unavailable")
        except RuntimeQueueError as exc:
            trace_runtime_event(
                "external_request_failed",
                correlation_id=correlation_id,
                outcome="failed",
                error_category=error_category(exc),
            )
            return _error(500, "workflow_failed", "Editorial workflow failed")
        trace_runtime_event(
            "external_request_completed",
            correlation_id=correlation_id,
            outcome="completed",
        )
        return HttpResponse(200, {"status": "completed", "result": result})


class ExternalBriefHttpServer(ThreadingHTTPServer):
    """Standard-library HTTP server bound to the application's asyncio loop."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        adapter: ExternalBriefHttpAdapter,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.adapter = adapter
        self.loop = loop
        super().__init__(address, ExternalBriefRequestHandler)


class ExternalBriefRequestHandler(BaseHTTPRequestHandler):
    """Minimal POST /brief transport bridge."""

    server: ExternalBriefHttpServer

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/brief":
            self._send(_error(404, "not_found", "Endpoint not found"))
            return
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._send(_error(400, "invalid_request", "Request body is invalid"))
            return
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            self._send(_error(400, "invalid_request", "Request body is invalid"))
            return
        body = self.rfile.read(length)
        future = asyncio.run_coroutine_threadsafe(
            self.server.adapter.handle(headers=self.headers, body=body),
            self.server.loop,
        )
        try:
            response = future.result()
        except Exception:
            response = _error(500, "internal_error", "Request could not be completed")
        self._send(response)

    def _send(self, response: HttpResponse) -> None:
        encoded = json.dumps(response.body, separators=(",", ":")).encode()
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args
