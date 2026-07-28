"""Model-backed Editor implementation."""

from editorial_team.agents.parsing import execute_text
from editorial_team.agents.prompts import editor_prompt
from editorial_team.domain.editorial import CriticReport, WritingTask
from editorial_team.models import ModelClient


class LlmEditor:
    """Apply one Critic report to an exact Writer output."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def revise(
        self,
        task: WritingTask,
        draft: str,
        report: CriticReport,
    ) -> str:
        """Return only the model's nonblank revised draft."""

        return execute_text(
            self._model,
            editor_prompt(task, draft, report),
            "Editor",
        )
