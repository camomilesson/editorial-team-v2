import pytest

from editorial_team.models import (
    FakeModelClient,
    ModelClientError,
    ModelRequest,
    ModelResponse,
    StructuredOutputSpec,
    ToolCall,
    ToolResult,
)


def test_fake_model_returns_scripted_responses_in_order_and_records_requests() -> None:
    first = ModelResponse("first", (), "one")
    second = ModelResponse("second", (), None)
    client = FakeModelClient([first, second])
    request = ModelRequest("hello")

    assert client.respond(request) is first
    assert client.respond(request) is second
    assert client.requests == [request, request]


def test_fake_model_reports_exhaustion() -> None:
    client = FakeModelClient([])

    with pytest.raises(ModelClientError, match="no scripted responses left"):
        client.respond(ModelRequest("hello"))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelRequest(""), "input"),
        (lambda: ModelRequest(()), "input"),
        (lambda: ModelRequest("hello", tools=[]), "tools"),
        (lambda: ModelResponse(text=1, tool_calls=(), continuation_token=None), "text"),
        (lambda: ModelResponse("", [], None), "tool_calls"),
        (lambda: ToolCall("", "lookup", {}), "call_id"),
        (lambda: ToolCall("1", "", {}), "name"),
        (lambda: ToolCall("1", "lookup", []), "arguments"),
        (lambda: ToolResult("1", "lookup", float("nan")), "JSON-compatible"),
    ],
)
def test_model_contracts_reject_invalid_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_request_accepts_tool_results_and_response_accepts_tool_calls() -> None:
    result = ToolResult("call-1", "lookup", {"value": 3})
    call = ToolCall("call-2", "save", {"value": 3})

    assert ModelRequest((result,), continuation_token="prior").input == (result,)
    assert ModelResponse("", (call,), "current").tool_calls == (call,)


def test_request_defaults_to_plain_text_output() -> None:
    first = ModelRequest("hello")
    second = ModelRequest("again")

    assert first.structured_output is None
    assert second.structured_output is None


def test_request_accepts_defensively_copied_structured_output() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    spec = StructuredOutputSpec("application/json", schema)
    request = ModelRequest("hello", structured_output=spec)

    schema["properties"]["value"]["type"] = "number"
    exposed = spec.schema
    exposed["properties"]["value"]["type"] = "boolean"

    assert request.structured_output is spec
    assert spec.schema["properties"]["value"]["type"] == "string"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: StructuredOutputSpec(" ", {"type": "object"}), "mime_type"),
        (lambda: StructuredOutputSpec("application/json", []), "schema"),
        (
            lambda: StructuredOutputSpec(
                "application/json",
                {"not_json": object()},
            ),
            "JSON-compatible",
        ),
        (
            lambda: ModelRequest("hello", structured_output=object()),
            "structured_output",
        ),
    ],
)
def test_structured_output_contract_rejects_invalid_values(
    factory: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
