import pytest

from editorial_team.models import ToolCall
from editorial_team.registry import ToolOutputError, ToolRegistry, ToolSpec

SCHEMA = {
    "name": "add",
    "description": "Add two integers.",
    "parameters": {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    },
}


def test_registers_and_dispatches_tool() -> None:
    registry = ToolRegistry(
        [ToolSpec(SCHEMA, lambda left, right: {"ok": True, "data": left + right})]
    )

    assert registry.names == ("add",)
    assert registry.schemas == (SCHEMA,)
    assert registry.execute(ToolCall("1", "add", {"left": 2, "right": 3})) == {
        "ok": True,
        "data": 5,
    }


def test_rejects_duplicate_tool_names() -> None:
    spec = ToolSpec(SCHEMA, lambda **_: {"ok": True, "data": None})

    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry([spec, spec])


def test_unknown_tool_returns_structured_error() -> None:
    output = ToolRegistry([]).execute(ToolCall("1", "missing", {}))

    assert output == {
        "ok": False,
        "error": {"type": "unknown_tool", "message": "Unknown tool: missing"},
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"left": 1},
        {"left": "1", "right": 2},
        {"left": 1, "right": 2, "extra": True},
    ],
)
def test_json_schema_validation_prevents_dispatch(arguments: dict[str, object]) -> None:
    called = False

    def handler(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"ok": True, "data": None}

    output = ToolRegistry([ToolSpec(SCHEMA, handler)]).execute(
        ToolCall("1", "add", arguments)
    )

    assert output["ok"] is False
    assert output["error"]["type"] == "invalid_tool_arguments"
    assert called is False


@pytest.mark.parametrize(
    "output",
    [
        None,
        {},
        {"ok": True},
        {"ok": False},
        {"ok": False, "error": "bad"},
        {"ok": True, "data": float("nan")},
    ],
)
def test_rejects_malformed_tool_output(output: object) -> None:
    registry = ToolRegistry([ToolSpec(SCHEMA, lambda **_: output)])

    with pytest.raises(ToolOutputError):
        registry.execute(ToolCall("1", "add", {"left": 1, "right": 2}))
