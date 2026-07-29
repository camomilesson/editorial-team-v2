"""Model-backed Critic implementation."""

from editorial_team.agents.parsing import execute_text, parse_critic_report
from editorial_team.agents.prompts import critic_prompt
from editorial_team.agents.schemas import CRITIC_STRUCTURED_OUTPUT
from editorial_team.domain.editorial import CriticReport, WritingTask
from editorial_team.models import ModelClient


class LlmCritic:
    """Review an exact draft and return a grounded structured report."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def review(self, task: WritingTask, draft: str) -> CriticReport:
        """Return a validated report for the exact supplied draft."""

        text = execute_text(
            self._model,
            critic_prompt(task, draft),
            "Critic",
            structured_output=CRITIC_STRUCTURED_OUTPUT,
        )
        return parse_critic_report(text, draft)
