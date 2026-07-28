"""Provider-neutral boundaries used by the conversation service."""

from __future__ import annotations

from typing import Protocol

from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import EditorialResult, WritingTask
from editorial_team.domain.routing import CoordinatorDecision


class Coordinator(Protocol):
    """Select one route for an incoming user message."""

    def decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        """Return a validated routing decision."""
        ...


class Talker(Protocol):
    """Produce an ordinary conversational response."""

    def respond(self, state: ConversationState, user_message: Message) -> str:
        """Return assistant text for the current user message."""
        ...


class WritingWorkflowRunner(Protocol):
    """Execute one complete writing cycle."""

    def execute(self, task: WritingTask) -> EditorialResult:
        """Return the result of the writing cycle."""
        ...


class ConversationStateStore(Protocol):
    """Load and save immutable conversation state."""

    def load(self, conversation_id: str) -> ConversationState | None:
        """Return stored state, if any."""
        ...

    def save(self, state: ConversationState) -> None:
        """Store a completed conversation state."""
        ...
