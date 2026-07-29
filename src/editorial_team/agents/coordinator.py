"""Model-backed Coordinator implementation."""

from editorial_team.agents.parsing import execute_text, parse_coordinator_decision
from editorial_team.agents.prompts import coordinator_prompt
from editorial_team.agents.schemas import COORDINATOR_STRUCTURED_OUTPUT
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.routing import CoordinatorDecision
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
