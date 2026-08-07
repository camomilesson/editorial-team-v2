"""Model-backed Coordinator implementations."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from editorial_team.agents.parsing import execute_text, parse_coordinator_decision
from editorial_team.agents.prompts import coordinator_prompt
from editorial_team.agents.schemas import COORDINATOR_STRUCTURED_OUTPUT
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.routing import CoordinatorDecision
from editorial_team.mlflow_tracing import (
    langchain_llm_span,
    mark_llm_failure,
    record_langchain_usage,
)
from editorial_team.models import ModelClient


class LlmCoordinator:
    """Classify one user message using a provider-neutral model client."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        """Return one validated route without user-facing prose."""

        text = execute_text(
            self._model,
            coordinator_prompt(state, user_message),
            "Coordinator",
            structured_output=COORDINATOR_STRUCTURED_OUTPUT,
        )
        return parse_coordinator_decision(text)


class ToolCallingCoordinator:
    """Invoke one LangChain chat-model step with exactly the scoped retrieval tools."""

    def __init__(self, chat_model: Any) -> None:
        if not callable(getattr(chat_model, "bind_tools", None)):
            raise ValueError("chat_model must support bind_tools")
        self._chat_model = chat_model

    def respond(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
    ) -> AIMessage:
        """Return one tool-call or final-decision AI message."""

        with langchain_llm_span(model=self._chat_model) as span:
            try:
                response = self._chat_model.bind_tools(list(tools)).invoke(list(messages))
                record_langchain_usage(span, response)
            except Exception:
                mark_llm_failure(span)
                raise RuntimeError("Coordinator model call failed") from None
        if not isinstance(response, AIMessage):
            raise RuntimeError("Coordinator returned invalid output")
        return response


def ai_message_text(message: AIMessage) -> str:
    """Extract ordered usable text without changing the provider message."""

    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") in {"text", "output_text"}
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        text = "".join(parts)
    else:
        text = ""
    if not text.strip():
        raise RuntimeError("Coordinator returned invalid output")
    return text
