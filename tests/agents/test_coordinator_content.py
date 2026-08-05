"""Regression tests for LangChain Coordinator final-message text extraction."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage

from editorial_team.agents.coordinator import ai_message_text
from editorial_team.agents.prompts import retrieval_coordinator_prompt
from editorial_team.domain.conversation import ConversationState, Message, MessageRole
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)


def test_extracts_ordinary_string_content() -> None:
    assert ai_message_text(AIMessage(content='{"route":"chat"}')) == '{"route":"chat"}'


def test_extracts_gemini_text_block_without_mutating_signature_metadata() -> None:
    content = [
        {
            "type": "text",
            "text": '{"route":"chat"}',
            "extras": {"signature": "provider-signature"},
        }
    ]
    message = AIMessage(content=content)

    assert ai_message_text(message) == '{"route":"chat"}'
    assert message.content == content
    assert message.content[0]["extras"]["signature"] == "provider-signature"  # type: ignore[index]


def test_concatenates_several_text_blocks_in_order() -> None:
    message = AIMessage(
        content=[
            {"type": "text", "text": '{"route":'},
            {"type": "text", "text": '"chat"}'},
        ]
    )
    assert ai_message_text(message) == '{"route":"chat"}'


def test_ignores_non_text_blocks_and_extracts_provider_output_text() -> None:
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "not model output"},
            {"type": "output_text", "text": "usable"},
            {"type": "image", "url": "https://example.invalid/image.png"},
        ]
    )
    assert ai_message_text(message) == "usable"


@pytest.mark.parametrize(
    "content",
    [[], [{"type": "thinking", "thinking": "internal"}], [{"type": "text", "text": " "}]],
)
def test_no_usable_text_fails_safely(content: list[dict[str, str]]) -> None:
    with pytest.raises(RuntimeError, match="invalid output"):
        ai_message_text(AIMessage(content=content))


def test_retrieval_coordinator_prompt_defines_exact_route_specific_absence() -> None:
    now = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    state = ConversationState("conversation-1")
    message = Message("message-1", "conversation-1", MessageRole.USER, "hello", now)

    prompt = retrieval_coordinator_prompt(
        state,
        message,
        current_local_datetime="2026-08-05T14:00:00+02:00",
        user_timezone="Europe/Madrid",
        current_utc_datetime="2026-08-05T12:00:00+00:00",
    )

    assert "chat for ordinary conversation has task_input=null" in prompt
    assert "Never use talker_context for a greeting explanation" in prompt
    assert "revise_task has a non-empty revision_instructions" in prompt
    assert "Use JSON null, not an empty string" in prompt


def test_retrieval_prompt_contrasts_active_and_historical_work_with_bounded_draft() -> None:
    now = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    task = WritingTask(
        "task-1",
        "conversation-1",
        WritingBrief("Write a formal Skyrim dragons post"),
        WritingTaskStatus.REVIEWED,
        now,
        now,
        "S" * 1300,
        CriticReport(CriticVerdict.PASS, "Approved"),
    )
    state = ConversationState("conversation-1", active_task=task)
    message = Message(
        "message-1",
        "conversation-1",
        MessageRole.USER,
        "Remember Aurora and add emojis",
        now,
    )

    prompt = retrieval_coordinator_prompt(
        state,
        message,
        current_local_datetime="2026-08-05T14:00:00+02:00",
        user_timezone="Europe/Madrid",
        current_utc_datetime="2026-08-05T12:00:00+00:00",
    )

    assert "explicit historical reference takes precedence over the active task" in prompt
    assert "Once you call search_corpus in a turn, never finish with revise_task" in prompt
    assert "prefer_recent=true" in prompt
    assert "Never copy, rewrite, summarize, or embed" in prompt
    assert "do not generate proposed replacement prose" in prompt
    assert "show_retrieved_draft" in prompt
    assert "inspection verbs do not imply editing" in prompt
    assert "Active task apples + user 'Make the sun" in prompt
    assert "imperative word 'make' does not override the named historical subject" in prompt
    assert '"original_request": "Write a formal Skyrim dragons post"' in prompt
    assert "S" * 1200 in prompt
    assert "S" * 1201 not in prompt
