"""Gemini implementation of the model client boundary."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

from editorial_team.mlflow_tracing import (
    gemini_llm_span,
    mark_llm_failure,
    record_gemini_usage,
)
from editorial_team.models import (
    ModelClientError,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
)

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


class GeminiModelClient:
    """Model client backed by the Gemini Interactions API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        sdk_client: Any | None = None,
        temperature: float | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A Gemini model name is required")

        self.model = model
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 < temperature <= 2
        ):
            raise ValueError("temperature must be greater than zero and at most two")
        self.temperature = temperature

        if sdk_client is not None:
            self._client = sdk_client
            return

        if not api_key:
            raise ValueError("A Gemini API key is required")

        self._client = genai.Client(api_key=api_key)

    def respond(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._convert_input(request.input),
        }

        if request.tools:
            kwargs["tools"] = list(request.tools)

        if self.temperature is not None:
            kwargs["generation_config"] = {"temperature": self.temperature}

        if request.continuation_token:
            kwargs["previous_interaction_id"] = request.continuation_token

        if request.structured_output is not None:
            kwargs["response_format"] = {
                "type": "text",
                "mime_type": request.structured_output.mime_type,
                "schema_": request.structured_output.schema,
            }

        with gemini_llm_span(model=self.model) as span:
            try:
                interaction = self._client.interactions.create(**kwargs)
                tool_calls = tuple(
                    ToolCall(
                        call_id=step.id,
                        name=step.name,
                        arguments=dict(step.arguments or {}),
                    )
                    for step in interaction.steps
                    if step.type == "function_call"
                )
                response = ModelResponse(
                    text=interaction.output_text or "",
                    tool_calls=tool_calls,
                    continuation_token=interaction.id,
                )
                record_gemini_usage(span, interaction)
                if span is not None:
                    span.set_attributes(
                        {
                            "response.text_present": bool(response.text),
                            "response.tool_call_count": len(response.tool_calls),
                        }
                    )
                return response
            except Exception:
                mark_llm_failure(span)
                raise ModelClientError("Gemini model call failed") from None

    @staticmethod
    def _convert_input(
        model_input: str | tuple[ToolResult, ...],
    ) -> str | list[dict[str, Any]]:
        if isinstance(model_input, str):
            return model_input

        return [
            {
                "type": "function_result",
                "name": result.name,
                "call_id": result.call_id,
                "result": [
                    {
                        "type": "text",
                        "text": json.dumps(result.result, allow_nan=False),
                    }
                ],
            }
            for result in model_input
        ]


def create_gemini_client_from_env(*, temperature: float | None = None) -> GeminiModelClient:
    """Create a Gemini client using environment variables."""

    provider = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()

    if provider != "gemini":
        raise ValueError(f"Unsupported MODEL_PROVIDER {provider!r}; expected 'gemini'")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    model = os.getenv("AGENT_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    return GeminiModelClient(model=model, api_key=api_key, temperature=temperature)


def create_gemini_chat_model_from_env(
    *, temperature: float | None = None
) -> ChatGoogleGenerativeAI:
    """Create the LangChain chat adapter with the same configured Gemini model."""

    provider = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()
    if provider != "gemini":
        raise ValueError(f"Unsupported MODEL_PROVIDER {provider!r}; expected 'gemini'")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    model = os.getenv("AGENT_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)
