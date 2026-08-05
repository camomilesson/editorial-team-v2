"""Provider-neutral boundaries used by the conversation service."""

from __future__ import annotations

from typing import Protocol

from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.routing import CoordinatorDecision, TalkerContext


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

    def respond(
        self,
        state: ConversationState,
        user_message: Message,
        context: TalkerContext | None = None,
    ) -> str:
        """Return assistant text for the current user message."""
        ...
