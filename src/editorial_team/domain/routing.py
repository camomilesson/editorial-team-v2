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
    REVISE_TASK = "revise_task"
    SHOW_RETRIEVED_DRAFT = "show_retrieved_draft"


class ClarificationReason(StrEnum):
    """Bounded reasons for retrieval-aware conversational clarification."""

    AMBIGUOUS = "ambiguous_candidates"
    NO_MATCH = "no_match"
    UNSUPPORTED_VERSION = "unsupported_relative_version"
    TOOL_PROBLEM = "tool_problem"


@dataclass(frozen=True)
class TalkerContext:
    """Small untrusted-data hint used by Talker for retrieval clarification."""

    reason: ClarificationReason
    candidate_summaries: tuple[str, ...]
    recommended_question: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, ClarificationReason):
            raise ValueError("reason must be a ClarificationReason")
        if not isinstance(self.candidate_summaries, tuple) or len(self.candidate_summaries) > 5:
            raise ValueError("candidate_summaries must contain at most five strings")
        object.__setattr__(
            self,
            "candidate_summaries",
            tuple(
                require_non_blank(value, "candidate_summary")[:600]
                for value in self.candidate_summaries
            ),
        )
        object.__setattr__(
            self,
            "recommended_question",
            require_non_blank(self.recommended_question, "recommended_question")[:600],
        )


@dataclass(frozen=True)
class CoordinatorDecision:
    """Validated route and optional route-specific input."""

    route: CoordinatorRoute
    confidence: float
    task_input: str | None = None
    revision_instructions: str | None = None
    talker_context: TalkerContext | None = None

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
        if self.talker_context is not None and not isinstance(self.talker_context, TalkerContext):
            raise ValueError("talker_context must be a TalkerContext")

        if self.route is CoordinatorRoute.START_WRITING_TASK:
            if self.task_input is None:
                raise ValueError("start_writing_task requires task_input")
            if self.revision_instructions is not None:
                raise ValueError("start_writing_task must not contain revision_instructions")
            if self.talker_context is not None:
                raise ValueError("start_writing_task must not contain talker_context")
        elif self.route is CoordinatorRoute.REVISE_TASK:
            if self.revision_instructions is None:
                raise ValueError("revise_task requires revision_instructions")
            if self.task_input is not None:
                raise ValueError("revise_task must not contain task_input")
            if self.talker_context is not None:
                raise ValueError("revise_task must not contain talker_context")
        elif self.route is CoordinatorRoute.SHOW_RETRIEVED_DRAFT:
            if self.task_input is not None or self.revision_instructions is not None:
                raise ValueError("show_retrieved_draft must not contain writing payloads")
            if self.talker_context is not None:
                raise ValueError("show_retrieved_draft must not contain talker_context")
        elif self.task_input is not None or self.revision_instructions is not None:
            raise ValueError("chat must not contain writing payloads")
