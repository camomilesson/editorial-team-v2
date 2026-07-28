"""Protocols for the participants in the writing workflow."""

from __future__ import annotations

from typing import Protocol

from editorial_team.domain.editorial import CriticReport, WritingTask


class Writer(Protocol):
    """Produce an initial draft for a writing task."""

    def write(self, task: WritingTask) -> str:
        """Return the first draft."""
        ...


class Critic(Protocol):
    """Review a draft against its writing task."""

    def review(self, task: WritingTask, draft: str) -> CriticReport:
        """Return a structured review of the exact supplied draft."""
        ...


class Editor(Protocol):
    """Revise a draft in response to a structured review."""

    def revise(
        self,
        task: WritingTask,
        draft: str,
        report: CriticReport,
    ) -> str:
        """Return one revised draft."""
        ...
