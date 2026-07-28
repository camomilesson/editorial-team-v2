"""Provider-independent model contracts."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

ToolSchema: TypeAlias = dict[str, Any]


class ModelClientError(RuntimeError):
    """Raised when a model client cannot produce a valid response."""


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_json_value(value: object, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool request made by a model."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.call_id, "call_id")
        _require_non_empty_string(self.name, "name")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be an object")
        _require_json_value(self.arguments, "arguments")


@dataclass(frozen=True)
class ToolResult:
    """A tool result to send back to the model."""

    call_id: str
    name: str
    result: Any

    def __post_init__(self) -> None:
        _require_non_empty_string(self.call_id, "call_id")
        _require_non_empty_string(self.name, "name")
        _require_json_value(self.result, "result")


ModelInput: TypeAlias = str | tuple[ToolResult, ...]


@dataclass(frozen=True)
class ModelRequest:
    """One request sent to a model."""

    input: ModelInput
    tools: tuple[ToolSchema, ...] = ()
    continuation_token: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.input, str):
            _require_non_empty_string(self.input, "input")
        elif not isinstance(self.input, tuple) or not self.input:
            raise ValueError("input must be a non-empty string or tuple of tool results")
        elif not all(isinstance(result, ToolResult) for result in self.input):
            raise ValueError("input must contain only ToolResult values")

        if not isinstance(self.tools, tuple) or not all(
            isinstance(schema, dict) for schema in self.tools
        ):
            raise ValueError("tools must be a tuple of object schemas")
        _require_json_value(self.tools, "tools")

        if self.continuation_token is not None:
            _require_non_empty_string(self.continuation_token, "continuation_token")


@dataclass(frozen=True)
class ModelResponse:
    """A normalized response returned by a model."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    continuation_token: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise ValueError("tool_calls must be a tuple of ToolCall values")
        if self.continuation_token is not None:
            _require_non_empty_string(self.continuation_token, "continuation_token")


class ModelClient(Protocol):
    """Interface implemented by model providers."""

    def respond(self, request: ModelRequest) -> ModelResponse:
        """Return one normalized model response."""
        ...


class FakeModelClient:
    """Return scripted responses for deterministic tests."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def respond(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise ValueError("request must be a ModelRequest")

        self.requests.append(request)

        if not self._responses:
            raise ModelClientError("Fake model has no scripted responses left")

        return self._responses.popleft()
