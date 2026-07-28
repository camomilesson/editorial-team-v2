"""Model-backed Talker implementation."""

from editorial_team.agents.parsing import execute_text
from editorial_team.agents.prompts import talker_prompt
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.models import ModelClient


class LlmTalker:
    """Produce one ordinary conversational response."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def respond(self, state: ConversationState, user_message: Message) -> str:
        """Return plain nonblank response text."""

        return execute_text(self._model, talker_prompt(state, user_message), "Talker")
