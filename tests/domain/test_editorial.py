from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone

import pytest

from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def minor_issue() -> CriticIssue:
    return CriticIssue(
        severity=CriticIssueSeverity.MINOR,
        location="Opening",
        problem="The opening is vague.",
        suggestion="Name the product.",
        grounded_excerpt="A new thing is here.",
    )


def passing_report() -> CriticReport:
    return CriticReport(
        verdict=CriticVerdict.PASS,
        summary="The draft meets the brief.",
        issues=(minor_issue(),),
    )


def revision_report() -> CriticReport:
    return CriticReport(
        verdict=CriticVerdict.REVISE,
        summary="The opening needs revision.",
        issues=(minor_issue(),),
    )


def make_task(
    status: WritingTaskStatus = WritingTaskStatus.CREATED,
    *,
    working_draft: object = None,
    critic_report: CriticReport | None = None,
) -> WritingTask:
    return WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=WritingBrief("Write a launch announcement."),
        status=status,
        created_at=NOW,
        updated_at=NOW,
        working_draft=working_draft,  # type: ignore[arg-type]
        critic_report=critic_report,
    )


def test_constructs_every_editorial_model() -> None:
    brief = WritingBrief(
        original_request="Write a launch announcement.",
        instructions=("Use a warm tone.", "Keep it concise."),
    )
    issue = minor_issue()
    report = passing_report()
    result = EditorialResult(
        writer_output="Canonical copy",
        critic_report=report,
        working_draft="Canonical copy",
        revision_applied=False,
    )
    task = WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=brief,
        status=WritingTaskStatus.REVIEWED,
        working_draft="Canonical copy",
        critic_report=report,
        created_at=NOW,
        updated_at=NOW,
    )

    assert issue.severity is CriticIssueSeverity.MINOR
    assert result.writer_output == result.working_draft
    assert task.brief.instructions == ("Use a warm tone.", "Keep it concise.")
    assert {status.value for status in WritingTaskStatus} == {
        "created",
        "drafted",
        "reviewed",
        "revised",
    }


def test_writing_task_has_only_one_draft_field() -> None:
    assert {field.name for field in fields(WritingTask)} == {
        "id",
        "conversation_id",
        "brief",
        "status",
        "created_at",
        "updated_at",
        "working_draft",
        "critic_report",
    }


def test_new_task_may_have_no_working_draft() -> None:
    assert make_task().working_draft is None


def test_present_working_draft_must_be_nonblank_text() -> None:
    assert make_task(working_draft="Existing copy").working_draft == "Existing copy"

    for value in ("", " ", 42):
        with pytest.raises(ValueError, match="working_draft"):
            make_task(working_draft=value)


@pytest.mark.parametrize(
    "status",
    [
        WritingTaskStatus.DRAFTED,
        WritingTaskStatus.REVIEWED,
        WritingTaskStatus.REVISED,
    ],
)
def test_produced_text_statuses_require_working_draft(status: WritingTaskStatus) -> None:
    with pytest.raises(ValueError, match="working_draft"):
        make_task(status)


@pytest.mark.parametrize(
    "status",
    [
        WritingTaskStatus.REVIEWED,
        WritingTaskStatus.REVISED,
    ],
)
def test_reviewed_statuses_require_critic_report(status: WritingTaskStatus) -> None:
    with pytest.raises(ValueError, match="critic_report"):
        make_task(status, working_draft="Copy")


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: WritingBrief(" "),
        lambda: WritingBrief("Request", (" ",)),
        lambda: CriticIssue(CriticIssueSeverity.MINOR, " "),
        lambda: CriticIssue(CriticIssueSeverity.MINOR, "Problem", location=" "),
        lambda: CriticReport(CriticVerdict.PASS, " "),
        lambda: EditorialResult(" ", passing_report(), "Copy", False),
        lambda: EditorialResult("Copy", passing_report(), " ", False),
    ],
)
def test_editorial_models_reject_blank_required_or_present_text(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_writing_task_validates_identifiers_timestamps_and_optional_text() -> None:
    common = {
        "brief": WritingBrief("Request"),
        "status": WritingTaskStatus.CREATED,
        "created_at": NOW,
        "updated_at": NOW,
    }

    with pytest.raises(ValueError, match="id"):
        WritingTask(" ", "conversation-1", **common)
    with pytest.raises(ValueError, match="conversation_id"):
        WritingTask("task-1", "../conversation", **common)
    with pytest.raises(ValueError, match="created_at"):
        WritingTask(
            "task-1",
            "conversation-1",
            WritingBrief("Request"),
            WritingTaskStatus.CREATED,
            datetime(2026, 7, 28),
            NOW,
        )
    with pytest.raises(ValueError, match="updated_at"):
        WritingTask(
            "task-1",
            "conversation-1",
            WritingBrief("Request"),
            WritingTaskStatus.CREATED,
            NOW,
            datetime(2026, 7, 28, tzinfo=timezone(timedelta(hours=1))),
        )
    with pytest.raises(ValueError, match="earlier"):
        WritingTask(
            "task-1",
            "conversation-1",
            WritingBrief("Request"),
            WritingTaskStatus.CREATED,
            NOW,
            NOW - timedelta(seconds=1),
        )


def test_passing_report_must_not_contain_major_issues() -> None:
    major = CriticIssue(CriticIssueSeverity.MAJOR, "The central claim is unsupported.")

    with pytest.raises(ValueError, match="must not contain major"):
        CriticReport(CriticVerdict.PASS, "Needs work.", (major,))


def test_revise_report_requires_at_least_one_issue() -> None:
    with pytest.raises(ValueError, match="at least one issue"):
        CriticReport(CriticVerdict.REVISE, "Changes are required.")

    assert revision_report().verdict is CriticVerdict.REVISE


def test_valid_pass_result_uses_writer_output_without_revision() -> None:
    report = passing_report()
    result = EditorialResult("Writer copy", report, "Writer copy", False)

    assert result.writer_output == "Writer copy"
    assert result.critic_report is report
    assert result.working_draft == "Writer copy"
    assert result.revision_applied is False


@pytest.mark.parametrize(
    ("working_draft", "revision_applied"),
    [("Writer copy", True), ("Different copy", False)],
)
def test_pass_result_rejects_inconsistent_revision_state(
    working_draft: str,
    revision_applied: bool,
) -> None:
    with pytest.raises(ValueError, match="passing result"):
        EditorialResult("Writer copy", passing_report(), working_draft, revision_applied)


def test_valid_revise_result_requires_revision_flag() -> None:
    report = revision_report()
    result = EditorialResult("Writer copy", report, "Editor copy", True)

    assert result.working_draft == "Editor copy"
    assert result.revision_applied is True

    with pytest.raises(ValueError, match="must apply a revision"):
        EditorialResult("Writer copy", report, "Editor copy", False)


def test_revise_result_does_not_require_textual_difference() -> None:
    result = EditorialResult("Copy", revision_report(), "Copy", True)
    assert result.writer_output == result.working_draft


@pytest.mark.parametrize("revision_applied", [0, 1, "yes", None])
def test_revision_applied_must_be_a_real_boolean(revision_applied: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        EditorialResult(
            "Writer copy",
            passing_report(),
            "Writer copy",
            revision_applied,  # type: ignore[arg-type]
        )
