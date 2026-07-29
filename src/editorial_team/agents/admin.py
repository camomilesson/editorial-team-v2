"""Capability-restricted model-backed operational AdminAgent."""

from editorial_team.agents.parsing import execute_text, parse_admin_assessment
from editorial_team.agents.prompts import admin_prompt
from editorial_team.agents.schemas import ADMIN_STRUCTURED_OUTPUT
from editorial_team.models import ModelClient
from editorial_team.operations.models import AdminAssessment, OperationalSnapshot
from editorial_team.operations.policy import AdminPolicy


class LlmAdminAgent:
    """Assess safe operational metadata with one structured model request."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def evaluate(
        self,
        snapshot: OperationalSnapshot,
        policy: AdminPolicy,
    ) -> AdminAssessment:
        """Return one structurally valid assessment without tools or retries."""

        text = execute_text(
            self._model,
            admin_prompt(snapshot, policy),
            "Admin",
            structured_output=ADMIN_STRUCTURED_OUTPUT,
        )
        return parse_admin_assessment(text)
