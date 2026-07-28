"""Generic tool registration, validation, and dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from jsonschema import Draft202012Validator

from editorial_team.models import ToolCall, ToolSchema

ToolOutput: TypeAlias = dict[str, Any]
ToolHandler: TypeAlias = Callable[..., ToolOutput]


class ToolOutputError(RuntimeError):
    """Raised when a tool handler returns malformed output."""


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool and the code that implements it."""

    schema: ToolSchema
    handler: ToolHandler

    @property
    def name(self) -> str:
        """Return the tool name."""

        name = self.schema.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Tool schema must contain a non-empty name")
        return name


class ToolRegistry:
    """Validate and execute model-requested tools."""

    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}

        for spec in specs:
            name = spec.name
            if name in self._specs:
                raise ValueError(f"Duplicate tool name: {name}")

            parameters = spec.schema.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(f"Tool {name} must define an object parameter schema")

            Draft202012Validator.check_schema(parameters)
            self._specs[name] = spec
            self._validators[name] = Draft202012Validator(parameters)

    @property
    def schemas(self) -> tuple[ToolSchema, ...]:
        """Return schemas that can be supplied to a model."""

        return tuple(spec.schema for spec in self._specs.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return all registered tool names."""

        return tuple(self._specs)

    def execute(self, tool_call: ToolCall) -> ToolOutput:
        """Validate and execute one normalized model tool call."""

        spec = self._specs.get(tool_call.name)
        if spec is None:
            return self._error(
                error_type="unknown_tool",
                message=f"Unknown tool: {tool_call.name}",
            )

        errors = sorted(
            self._validators[tool_call.name].iter_errors(tool_call.arguments),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            return self._error(
                error_type="invalid_tool_arguments",
                message="; ".join(error.message for error in errors),
            )

        result = spec.handler(**tool_call.arguments)
        self._validate_output(tool_name=tool_call.name, output=result)
        return result

    @staticmethod
    def _validate_output(*, tool_name: str, output: object) -> None:
        if not isinstance(output, dict):
            raise ToolOutputError(f"Tool {tool_name} returned a non-object result")

        ok = output.get("ok")
        if not isinstance(ok, bool):
            raise ToolOutputError(f"Tool {tool_name} result must contain boolean 'ok'")
        if ok and "data" not in output:
            raise ToolOutputError(f"Successful tool {tool_name} result must contain 'data'")
        if not ok and "error" not in output:
            raise ToolOutputError(f"Failed tool {tool_name} result must contain 'error'")

        if not ok:
            error = output["error"]
            if not isinstance(error, dict):
                raise ToolOutputError(
                    f"Failed tool {tool_name} result must contain an error object"
                )
            if not isinstance(error.get("type"), str):
                raise ToolOutputError(
                    f"Failed tool {tool_name} error must contain string 'type'"
                )
            if not isinstance(error.get("message"), str):
                raise ToolOutputError(
                    f"Failed tool {tool_name} error must contain string 'message'"
                )

        try:
            json.dumps(output, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolOutputError(
                f"Tool {tool_name} returned a non-JSON-serializable result"
            ) from exc

    @staticmethod
    def _error(*, error_type: str, message: str) -> ToolOutput:
        return {"ok": False, "error": {"type": error_type, "message": message}}
