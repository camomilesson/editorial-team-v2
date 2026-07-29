"""Conversation and message state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier
from editorial_team.domain.editorial import WritingTask


class MessageRole(StrEnum):
    """The participant responsible for a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """One message in a conversation."""

    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_identifier(self.id, "id"))
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        if not isinstance(self.role, MessageRole):
            raise ValueError("role must be a MessageRole")
        object.__setattr__(self, "content", require_non_blank(self.content, "content"))
        require_utc_timestamp(self.created_at, "created_at")


@dataclass(frozen=True)
class ConversationState:
    """Recent conversation context and its latest writing task."""

    conversation_id: str
    recent_messages: tuple[Message, ...] = ()
    active_task: WritingTask | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        if not isinstance(self.recent_messages, tuple) or not all(
            isinstance(message, Message) for message in self.recent_messages
        ):
            raise ValueError("recent_messages must be a tuple of Message values")
        if self.active_task is not None:
            if not isinstance(self.active_task, WritingTask):
                raise ValueError("active_task must be a WritingTask")
            if self.active_task.conversation_id != self.conversation_id:
                raise ValueError("active_task must belong to the conversation")
        if any(message.conversation_id != self.conversation_id for message in self.recent_messages):
            raise ValueError("recent_messages must belong to the conversation")
