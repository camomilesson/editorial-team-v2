"""Writing-task and editorial-review domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier


class WritingTaskStatus(StrEnum):
    """Lifecycle state of a writing task."""

    CREATED = "created"
    DRAFTED = "drafted"
    REVIEWED = "reviewed"
    REVISED = "revised"


@dataclass(frozen=True)
class WritingBrief:
    """Small, provider-neutral input for a writing task."""

    original_request: str
    instructions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_request",
            require_non_blank(self.original_request, "original_request"),
        )
        if not isinstance(self.instructions, tuple):
            raise ValueError("instructions must be a tuple of strings")
        object.__setattr__(
            self,
            "instructions",
            tuple(
                require_non_blank(instruction, f"instructions[{index}]")
                for index, instruction in enumerate(self.instructions)
            ),
        )


class CriticVerdict(StrEnum):
    """Outcome of an editorial review."""

    PASS = "pass"
    REVISE = "revise"


class CriticIssueSeverity(StrEnum):
    """Impact of an issue identified during review."""

    MINOR = "minor"
    MAJOR = "major"


@dataclass(frozen=True)
class CriticIssue:
    """One concrete issue identified in a draft."""

    severity: CriticIssueSeverity
    problem: str
    location: str | None = None
    suggestion: str | None = None
    grounded_excerpt: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, CriticIssueSeverity):
            raise ValueError("severity must be a CriticIssueSeverity")
        object.__setattr__(self, "problem", require_non_blank(self.problem, "problem"))
        for field_name in ("location", "suggestion", "grounded_excerpt"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_non_blank(value, field_name))


@dataclass(frozen=True)
class CriticReport:
    """Structured review of a draft."""

    verdict: CriticVerdict
    summary: str
    issues: tuple[CriticIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, CriticVerdict):
            raise ValueError("verdict must be a CriticVerdict")
        object.__setattr__(self, "summary", require_non_blank(self.summary, "summary"))
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, CriticIssue) for issue in self.issues
        ):
            raise ValueError("issues must be a tuple of CriticIssue values")
        if self.verdict is CriticVerdict.PASS and any(
            issue.severity is CriticIssueSeverity.MAJOR for issue in self.issues
        ):
            raise ValueError("a passing critic report must not contain major issues")
        if self.verdict is CriticVerdict.REVISE and not self.issues:
            raise ValueError("a revise critic report must contain at least one issue")


@dataclass(frozen=True)
class EditorialResult:
    """Final output and review details for one editorial attempt."""

    writer_output: str
    critic_report: CriticReport
    working_draft: str
    revision_applied: bool

    def __post_init__(self) -> None:
        require_non_blank(self.writer_output, "writer_output")
        if not isinstance(self.critic_report, CriticReport):
            raise ValueError("critic_report must be a CriticReport")
        require_non_blank(self.working_draft, "working_draft")
        if not isinstance(self.revision_applied, bool):
            raise ValueError("revision_applied must be a boolean")
        if self.critic_report.verdict is CriticVerdict.PASS:
            if self.revision_applied:
                raise ValueError("a passing result must not apply a revision")
            if self.working_draft != self.writer_output:
                raise ValueError("a passing result must use writer_output as working_draft")
        elif self.critic_report.verdict is CriticVerdict.REVISE:
            if not self.revision_applied:
                raise ValueError("a revise result must apply a revision")
        else:
            raise ValueError("critic_report must contain a supported verdict")


@dataclass(frozen=True)
class WritingTask:
    """State accumulated for one writing request."""

    id: str
    conversation_id: str
    brief: WritingBrief
    status: WritingTaskStatus
    created_at: datetime
    updated_at: datetime
    working_draft: str | None = None
    critic_report: CriticReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_identifier(self.id, "id"))
        object.__setattr__(
            self,
            "conversation_id",
            validate_identifier(self.conversation_id, "conversation_id"),
        )
        if not isinstance(self.brief, WritingBrief):
            raise ValueError("brief must be a WritingBrief")
        if not isinstance(self.status, WritingTaskStatus):
            raise ValueError("status must be a WritingTaskStatus")
        require_utc_timestamp(self.created_at, "created_at")
        require_utc_timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        if self.working_draft is not None:
            require_non_blank(self.working_draft, "working_draft")
        if self.critic_report is not None and not isinstance(self.critic_report, CriticReport):
            raise ValueError("critic_report must be a CriticReport")
        produced_text_statuses = {
            WritingTaskStatus.DRAFTED,
            WritingTaskStatus.REVIEWED,
            WritingTaskStatus.REVISED,
        }
        reviewed_statuses = {
            WritingTaskStatus.REVIEWED,
            WritingTaskStatus.REVISED,
        }
        if self.status in produced_text_statuses and self.working_draft is None:
            raise ValueError(f"{self.status.value} tasks require working_draft")
        if self.status in reviewed_statuses and self.critic_report is None:
            raise ValueError(f"{self.status.value} tasks require critic_report")
