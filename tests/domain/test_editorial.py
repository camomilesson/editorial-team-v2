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


def test_constructs_every_editorial_model() -> None:
    brief = WritingBrief(
        original_request="Write a launch announcement.",
        instructions=("Use a warm tone.", "Keep it concise."),
    )
    issue = minor_issue()
    report = passing_report()
    result = EditorialResult(
        first_draft="Final copy",
        critic_report=report,
        final_draft="Final copy",
    )
    task = WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=brief,
        status=WritingTaskStatus.AWAITING_USER_EVALUATION,
        first_draft="First copy",
        critic_report=report,
        updated_draft="Final copy",
        final_draft="Final copy",
        user_evaluation="Please make the title shorter.",
        created_at=NOW,
        updated_at=NOW,
    )

    assert issue.severity is CriticIssueSeverity.MINOR
    assert result.updated_draft is None
    assert task.brief.instructions == ("Use a warm tone.", "Keep it concise.")
    assert {status.value for status in WritingTaskStatus} == {
        "created",
        "drafted",
        "reviewed",
        "revised",
        "awaiting_user_evaluation",
        "approved",
    }


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: WritingBrief(" "),
        lambda: WritingBrief("Request", (" ",)),
        lambda: CriticIssue(CriticIssueSeverity.MINOR, " "),
        lambda: CriticIssue(CriticIssueSeverity.MINOR, "Problem", location=" "),
        lambda: CriticReport(CriticVerdict.PASS, " "),
        lambda: EditorialResult(" ", passing_report(), "Final"),
        lambda: EditorialResult("Draft", passing_report(), " "),
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
    with pytest.raises(ValueError, match="user_evaluation"):
        WritingTask("task-1", "conversation-1", user_evaluation=" ", **common)


def test_passing_report_must_not_contain_major_issues() -> None:
    major = CriticIssue(CriticIssueSeverity.MAJOR, "The central claim is unsupported.")

    with pytest.raises(ValueError, match="must not contain major"):
        CriticReport(CriticVerdict.PASS, "Needs work.", (major,))


def test_revise_report_requires_at_least_one_issue() -> None:
    with pytest.raises(ValueError, match="at least one issue"):
        CriticReport(CriticVerdict.REVISE, "Changes are required.")

    report = CriticReport(CriticVerdict.REVISE, "Clarify the opening.", (minor_issue(),))
    assert report.verdict is CriticVerdict.REVISE


def test_editorial_result_without_update_uses_first_draft_as_final() -> None:
    report = passing_report()

    assert EditorialResult("Draft", report, "Draft").final_draft == "Draft"
    with pytest.raises(ValueError, match="equal first_draft"):
        EditorialResult("Draft", report, "Different")


def test_editorial_result_with_update_uses_updated_draft_as_final() -> None:
    report = CriticReport(CriticVerdict.REVISE, "Revise.", (minor_issue(),))

    result = EditorialResult("Draft", report, "Updated", updated_draft="Updated")
    assert result.final_draft == result.updated_draft

    with pytest.raises(ValueError, match="equal updated_draft"):
        EditorialResult("Draft", report, "Different", updated_draft="Updated")
