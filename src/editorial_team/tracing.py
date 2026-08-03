"""Sanitized context-local tracing for live turns."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

trace_logger = logging.getLogger("editorial_team.live_trace")

_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_trace_context: ContextVar[TurnTrace | None] = ContextVar("turn_trace", default=None)


@dataclass
class TurnTrace:
    """Safe per-turn metadata shared across synchronous boundaries."""

    correlation_id: str
    update_id: int | None
    stage: str = "telegram"


def trace_for_update(update_id: int | None) -> TurnTrace:
    """Create a correlation-safe trace without chat or user data."""

    if isinstance(update_id, int) and not isinstance(update_id, bool):
        encoded = f"n{abs(update_id)}" if update_id < 0 else str(update_id)
        correlation_id = f"tg-{encoded}"
    else:
        correlation_id = f"tg-{uuid4().hex[:12]}"
    return TurnTrace(correlation_id=correlation_id, update_id=update_id)


@contextmanager
def bind_turn_trace(trace: TurnTrace):
    """Bind one trace to the current async or synchronous execution context."""

    token = _trace_context.set(trace)
    try:
        yield trace
    finally:
        _trace_context.reset(token)


def set_trace_stage(stage: str) -> None:
    """Record the narrowest currently executing safe stage."""

    trace = _trace_context.get()
    if trace is not None:
        trace.stage = _safe(stage)


def current_trace_stage() -> str:
    """Return the current safe stage, or an application fallback."""

    trace = _trace_context.get()
    return "application" if trace is None else trace.stage


def trace_event(event: str, **fields: Any) -> None:
    """Log one structured event containing only explicitly safe scalar metadata."""

    trace = _trace_context.get()
    if trace is None:
        return
    parts = [
        _safe(event),
        f"correlation_id={_safe(trace.correlation_id)}",
        f"update_id={_safe(trace.update_id)}",
    ]
    for key, value in fields.items():
        parts.append(f"{_safe(key)}={_safe(value)}")
    trace_logger.info(" ".join(parts))


def trace_runtime_event(event: str, *, correlation_id: str, **fields: Any) -> None:
    """Log one queue event using only explicit safe scalar metadata."""

    parts = [
        _safe(event),
        f"correlation_id={_safe(correlation_id)}",
    ]
    for key, value in fields.items():
        parts.append(f"{_safe(key)}={_safe(value)}")
    trace_logger.info(" ".join(parts))


def error_category(error: BaseException) -> str:
    """Return a sanitized category without exposing exception text."""

    if type(error).__name__ == "AgentError" and type(error).__module__ == (
        "editorial_team.agents.errors"
    ):
        categories = {
            "Coordinator model call failed": "provider_model_failure",
            "Coordinator returned invalid output": "blank_response",
            "Model returned invalid JSON": "json_decoding_failure",
            "Model returned invalid JSON object": "json_decoding_failure",
            "Model returned unexpected structured-output fields": "schema_validation_failure",
            "Coordinator returned invalid structured output": "domain_consistency_failure",
            "Admin model call failed": "provider_model_failure",
            "Admin returned invalid output": "blank_response",
            "Admin returned invalid structured output": "domain_consistency_failure",
        }
        return categories.get(str(error), "agent_error")
    name = type(error).__name__
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return _safe(normalized)


def _safe(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not _SAFE_VALUE.fullmatch(text):
        raise ValueError("trace metadata must be a safe scalar")
    return text
