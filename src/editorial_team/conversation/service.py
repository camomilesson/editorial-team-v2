"""Thin application facade over the compiled conversation graph."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from editorial_team.contracts.common import require_non_blank
from editorial_team.contracts.identity import validate_identifier
from editorial_team.domain.conversation import Message
from editorial_team.errors import ServiceError
from editorial_team.mlflow_tracing import (
    RequestOrigin,
    agent_invocation_span,
    record_batch_candidate_answer,
    validate_request_origin,
)


class ConversationServiceError(ServiceError):
    """A sanitized failure at the conversation application boundary."""


class ConversationService:
    """Validate and invoke one already-compiled authoritative graph."""

    def __init__(
        self,
        *,
        graph_runner: Any,
        close_checkpointer: Callable[[], None] | None = None,
    ) -> None:
        if graph_runner is None or not callable(getattr(graph_runner, "invoke", None)):
            raise ValueError("graph_runner must provide invoke")
        self._graph_runner = graph_runner
        self._close_checkpointer = close_checkpointer

    def process_message(
        self,
        conversation_id: str,
        text: str,
        *,
        request_origin: RequestOrigin = "api",
        eval_case_id: str | None = None,
    ) -> tuple[Message, ...]:
        """Process one turn and return only its assistant messages."""

        try:
            conversation_id = validate_identifier(conversation_id, "conversation_id")
            text = require_non_blank(text, "text")
            request_origin = validate_request_origin(request_origin)
            if eval_case_id is not None:
                eval_case_id = validate_identifier(eval_case_id, "eval_case_id")
        except ValueError:
            raise ConversationServiceError("Invalid conversation input") from None
        with agent_invocation_span(
            request_origin=request_origin, eval_case_id=eval_case_id
        ) as root_span:
            try:
                graph_state = self._graph_runner.invoke(
                    {
                        "state_version": 1,
                        "invocation_kind": "conversation",
                        "conversation_id": conversation_id,
                        "input_text": text,
                    },
                    {"configurable": {"thread_id": f"editorial:v1:{conversation_id}"}},
                )
            except ServiceError as exc:
                raise ConversationServiceError(str(exc)) from None
            except Exception:
                raise ConversationServiceError("Conversation graph failed") from None
            messages = graph_state.get("assistant_messages")
            if (
                not isinstance(messages, tuple)
                or not messages
                or not all(isinstance(message, Message) for message in messages)
            ):
                raise ConversationServiceError("Conversation graph returned an invalid result")
            if root_span is not None:
                root_span.set_attribute("assistant_message_count", len(messages))
                record_batch_candidate_answer(root_span, messages)
            return messages

    def close(self) -> None:
        """Release the application-owned checkpoint resource once."""

        closer = self._close_checkpointer
        if closer is not None:
            self._close_checkpointer = None
            closer()
