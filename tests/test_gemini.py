from types import SimpleNamespace

import pytest

from editorial_team.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiModelClient,
    create_gemini_client_from_env,
)
from editorial_team.models import (
    ModelClientError,
    ModelRequest,
    StructuredOutputSpec,
    ToolResult,
)


class FakeInteractions:
    def __init__(self, interaction: object = None, error: Exception | None = None) -> None:
        self.interaction = interaction
        self.error = error
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.interaction


def sdk_with(interactions: FakeInteractions) -> SimpleNamespace:
    return SimpleNamespace(interactions=interactions)


def test_normalizes_text_and_function_calls() -> None:
    interaction = SimpleNamespace(
        id="interaction-1",
        output_text="Done",
        steps=[
            SimpleNamespace(type="text"),
            SimpleNamespace(
                type="function_call",
                id="call-1",
                name="lookup",
                arguments={"query": "example"},
            ),
        ],
    )
    interactions = FakeInteractions(interaction)
    client = GeminiModelClient(sdk_client=sdk_with(interactions))

    response = client.respond(
        ModelRequest(
            "Find it",
            tools=({"name": "lookup", "parameters": {"type": "object"}},),
            continuation_token="previous",
        )
    )

    assert response.text == "Done"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "example"}
    assert response.continuation_token == "interaction-1"
    assert interactions.kwargs == {
        "model": DEFAULT_GEMINI_MODEL,
        "input": "Find it",
        "tools": [{"name": "lookup", "parameters": {"type": "object"}}],
        "previous_interaction_id": "previous",
    }


def test_converts_tool_results_to_gemini_input() -> None:
    interaction = SimpleNamespace(id="next", output_text=None, steps=[])
    interactions = FakeInteractions(interaction)
    client = GeminiModelClient(sdk_client=sdk_with(interactions))

    response = client.respond(ModelRequest((ToolResult("call-1", "lookup", {"found": True}),)))

    assert response.text == ""
    assert interactions.kwargs["input"] == [
        {
            "type": "function_result",
            "name": "lookup",
            "call_id": "call-1",
            "result": [{"type": "text", "text": '{"found": true}'}],
        }
    ]


def test_evaluation_temperature_is_explicit_and_default_is_unchanged() -> None:
    default_interactions = FakeInteractions(SimpleNamespace(id="one", output_text="ok", steps=[]))
    GeminiModelClient(sdk_client=sdk_with(default_interactions)).respond(ModelRequest("hello"))
    assert "generation_config" not in default_interactions.kwargs

    eval_interactions = FakeInteractions(SimpleNamespace(id="two", output_text="ok", steps=[]))
    client = GeminiModelClient(sdk_client=sdk_with(eval_interactions), temperature=0.2)
    client.respond(ModelRequest("hello"))
    assert eval_interactions.kwargs["generation_config"] == {"temperature": 0.2}


def test_forwards_structured_output_in_interactions_format() -> None:
    interaction = SimpleNamespace(id="structured", output_text='{"ok":true}', steps=[])
    interactions = FakeInteractions(interaction)
    client = GeminiModelClient(sdk_client=sdk_with(interactions))
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    response = client.respond(
        ModelRequest(
            "Return JSON",
            tools=({"name": "lookup", "parameters": {"type": "object"}},),
            continuation_token="previous",
            structured_output=StructuredOutputSpec("application/json", schema),
        )
    )

    assert response.text == '{"ok":true}'
    assert interactions.kwargs == {
        "model": DEFAULT_GEMINI_MODEL,
        "input": "Return JSON",
        "tools": [{"name": "lookup", "parameters": {"type": "object"}}],
        "previous_interaction_id": "previous",
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema_": schema,
        },
    }


@pytest.mark.parametrize(
    "interactions",
    [
        FakeInteractions(error=RuntimeError("secret provider detail")),
        FakeInteractions(SimpleNamespace(id="bad", output_text="", steps=[object()])),
    ],
)
def test_sanitizes_provider_and_normalization_failures(interactions: FakeInteractions) -> None:
    client = GeminiModelClient(sdk_client=sdk_with(interactions))

    with pytest.raises(ModelClientError, match=r"^Gemini model call failed$") as caught:
        client.respond(ModelRequest("hello"))

    assert "secret provider detail" not in str(caught.value)


def test_factory_validates_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "other")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    with pytest.raises(ValueError, match="Unsupported MODEL_PROVIDER"):
        create_gemini_client_from_env()

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(ValueError, match="not configured"):
        create_gemini_client_from_env()
