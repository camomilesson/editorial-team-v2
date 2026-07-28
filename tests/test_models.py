import pytest

from editorial_team.models import (
    FakeModelClient,
    ModelClientError,
    ModelRequest,
    ModelResponse,
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
