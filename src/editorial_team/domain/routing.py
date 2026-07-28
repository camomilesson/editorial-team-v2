"""Coordinator routing decisions represented as domain data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from editorial_team.contracts.common import require_non_blank


class CoordinatorRoute(StrEnum):
    """Supported destinations for an incoming user message."""

    CHAT = "chat"
    START_WRITING_TASK = "start_writing_task"
    APPROVE_TASK = "approve_task"
    REVISE_TASK = "revise_task"


@dataclass(frozen=True)
class CoordinatorDecision:
    """Validated route and optional route-specific input."""

    route: CoordinatorRoute
    confidence: float
    task_input: str | None = None
    revision_instructions: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route, CoordinatorRoute):
            raise ValueError("route must be a CoordinatorRoute")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between 0 and 1")

        if self.task_input is not None:
            object.__setattr__(
                self,
                "task_input",
                require_non_blank(self.task_input, "task_input"),
            )
        if self.revision_instructions is not None:
            object.__setattr__(
                self,
                "revision_instructions",
                require_non_blank(self.revision_instructions, "revision_instructions"),
            )

        if self.route is CoordinatorRoute.START_WRITING_TASK:
            if self.task_input is None:
                raise ValueError("start_writing_task requires task_input")
            if self.revision_instructions is not None:
                raise ValueError("start_writing_task must not contain revision_instructions")
        elif self.route is CoordinatorRoute.REVISE_TASK:
            if self.revision_instructions is None:
                raise ValueError("revise_task requires revision_instructions")
            if self.task_input is not None:
                raise ValueError("revise_task must not contain task_input")
        elif self.task_input is not None or self.revision_instructions is not None:
            raise ValueError("chat and approve_task must not contain writing payloads")
