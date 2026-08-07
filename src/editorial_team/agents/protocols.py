"""Protocols for the participants in the writing workflow."""

from __future__ import annotations

from typing import Protocol

from editorial_team.domain.editorial import CriticReport, EditorialRunContext
from editorial_team.operations.models import AdminAssessment, OperationalSnapshot
from editorial_team.operations.policy import AdminPolicy


class AdminAgent(Protocol):
    """Evaluate only safe operational facts under an explicit policy."""

    def evaluate(
        self,
        snapshot: OperationalSnapshot,
        policy: AdminPolicy,
    ) -> AdminAssessment:
        """Return one structured operational assessment."""
        ...


class Writer(Protocol):
    """Produce an initial draft for a writing task."""

    def write(self, context: EditorialRunContext) -> str:
        """Return the first draft."""
        ...


class Critic(Protocol):
    """Review a draft against its writing task."""

    def review(self, context: EditorialRunContext, draft: str) -> CriticReport:
        """Return a structured review of the exact supplied draft."""
        ...


class Editor(Protocol):
    """Revise a draft in response to a structured review."""

    def revise(
        self,
        context: EditorialRunContext,
        draft: str,
        report: CriticReport,
    ) -> str:
        """Return one revised draft."""
        ...
