"""Model-backed Writer implementation."""

from editorial_team.agents.parsing import execute_text
from editorial_team.agents.prompts import writer_prompt
from editorial_team.domain.editorial import EditorialRunContext
from editorial_team.models import ModelClient


class LlmWriter:
    """Create or rewrite a draft from task context."""

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def write(self, context: EditorialRunContext) -> str:
        """Return only the model's nonblank draft text."""

        return execute_text(self._model, writer_prompt(context), "Writer")
