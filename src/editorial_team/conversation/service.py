"""Thin application facade over the compiled conversation graph."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from editorial_team.contracts.common import require_non_blank
from editorial_team.contracts.identity import validate_identifier
from editorial_team.domain.conversation import Message, MessageRole
from editorial_team.errors import ServiceError
from editorial_team.mlflow_tracing import (
    RequestOrigin,
    agent_invocation_span,
    record_batch_candidate_answer,
    validate_request_origin,
)
from editorial_team.safety import (
    ATTR_OUTPUT_REPLACED,
    ATTR_POSTFLIGHT_FLAGGED,
    SAFE_BLOCKED_RESPONSE,
    SAFE_OUTPUT_RESPONSE,
    detect_input_threat,
    filter_output,
    safety_attributes,
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
        protected_output_markers: tuple[str, ...] = (),
    ) -> None:
        if graph_runner is None or not callable(getattr(graph_runner, "invoke", None)):
            raise ValueError("graph_runner must provide invoke")
        self._graph_runner = graph_runner
        self._close_checkpointer = close_checkpointer
        self._protected_output_markers = tuple(
            marker for marker in protected_output_markers if isinstance(marker, str) and marker
        )

    def process_message(
        self,
        conversation_id: str,
        text: str,
        *,
        request_origin: RequestOrigin = "api",
        eval_case_id: str | None = None,
        eval_run_number: int | None = None,
        eval_agent_temperature: float | None = None,
    ) -> tuple[Message, ...]:
        """Process one turn and return only its assistant messages."""

        try:
            conversation_id = validate_identifier(conversation_id, "conversation_id")
            text = require_non_blank(text, "text")
            request_origin = validate_request_origin(request_origin)
            if eval_case_id is not None:
                eval_case_id = validate_identifier(eval_case_id, "eval_case_id")
            if eval_run_number is not None and (
                isinstance(eval_run_number, bool)
                or not isinstance(eval_run_number, int)
                or eval_run_number <= 0
            ):
                raise ValueError("eval_run_number must be positive")
            if eval_agent_temperature is not None and (
                isinstance(eval_agent_temperature, bool)
                or not isinstance(eval_agent_temperature, (int, float))
                or eval_agent_temperature <= 0
            ):
                raise ValueError("eval_agent_temperature must be positive")
        except ValueError:
            raise ConversationServiceError("Invalid conversation input") from None
        with agent_invocation_span(
            request_origin=request_origin,
            eval_case_id=eval_case_id,
            eval_run_number=eval_run_number,
            eval_agent_temperature=eval_agent_temperature,
        ) as root_span:
            preflight = detect_input_threat(text)
            if root_span is not None:
                root_span.set_attributes(
                    safety_attributes(preflight, input_blocked=preflight.flagged)
                )
            if preflight.flagged:
                return (_safe_message(conversation_id, SAFE_BLOCKED_RESPONSE),)
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
            postflight = filter_output(
                "\n\n".join(message.content for message in messages),
                protected_markers=self._protected_output_markers,
            )
            if postflight.flagged:
                messages = (_safe_message(conversation_id, SAFE_OUTPUT_RESPONSE),)
            if root_span is not None:
                postflight_attributes: dict[str, object] = {
                    ATTR_POSTFLIGHT_FLAGGED: postflight.flagged,
                    ATTR_OUTPUT_REPLACED: postflight.flagged,
                }
                if postflight.flagged:
                    postflight_attributes["safety.reason_codes"] = list(postflight.reason_codes)
                root_span.set_attributes(postflight_attributes)
                root_span.set_attribute("assistant_message_count", len(messages))
                record_batch_candidate_answer(root_span, messages)
            return messages

    def close(self) -> None:
        """Release the application-owned checkpoint resource once."""

        closer = self._close_checkpointer
        if closer is not None:
            self._close_checkpointer = None
            closer()


def _safe_message(conversation_id: str, content: str) -> Message:
    return Message(uuid4().hex, conversation_id, MessageRole.ASSISTANT, content, datetime.now(UTC))
