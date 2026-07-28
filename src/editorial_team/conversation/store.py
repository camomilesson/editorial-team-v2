"""In-memory conversation state storage."""

from __future__ import annotations

from copy import deepcopy

from editorial_team.domain.conversation import ConversationState


class InMemoryConversationStateStore:
    """Store defensive state copies without module-global sharing."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def load(self, conversation_id: str) -> ConversationState | None:
        """Return a defensive copy of one conversation."""

        state = self._states.get(conversation_id)
        return None if state is None else deepcopy(state)

    def save(self, state: ConversationState) -> None:
        """Store a defensive copy of a completed state."""

        self._states[state.conversation_id] = deepcopy(state)
