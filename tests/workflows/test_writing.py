from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.workflows.writing import WritingWorkflow, WritingWorkflowError

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def make_task() -> WritingTask:
    return WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=WritingBrief("Write a concise announcement."),
        status=WritingTaskStatus.CREATED,
        created_at=NOW,
        updated_at=NOW,
    )


def pass_report() -> CriticReport:
    return CriticReport(CriticVerdict.PASS, "The draft meets the brief.")


def revise_report() -> CriticReport:
    issue = CriticIssue(
        CriticIssueSeverity.MAJOR,
        "The benefit is unclear.",
        suggestion="State the benefit directly.",
    )
    return CriticReport(CriticVerdict.REVISE, "One revision is required.", (issue,))


@dataclass
class RecordingWriter:
    output: object
    calls: list[WritingTask] = field(default_factory=list)
    order: list[str] | None = None

    def write(self, task: WritingTask) -> Any:
        self.calls.append(task)
        if self.order is not None:
            self.order.append("writer")
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingCritic:
    output: object
    calls: list[tuple[WritingTask, str]] = field(default_factory=list)
    order: list[str] | None = None

    def review(self, task: WritingTask, draft: str) -> Any:
        self.calls.append((task, draft))
        if self.order is not None:
            self.order.append("critic")
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingEditor:
    output: object
    calls: list[tuple[WritingTask, str, CriticReport]] = field(default_factory=list)
    order: list[str] | None = None

    def revise(self, task: WritingTask, draft: str, report: CriticReport) -> Any:
        self.calls.append((task, draft, report))
        if self.order is not None:
            self.order.append("editor")
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def make_workflow(
    *,
    writer_output: object = "First draft",
    critic_output: object | None = None,
    editor_output: object = "Updated draft",
    order: list[str] | None = None,
) -> tuple[WritingWorkflow, RecordingWriter, RecordingCritic, RecordingEditor]:
    writer = RecordingWriter(writer_output, order=order)
    critic = RecordingCritic(
        pass_report() if critic_output is None else critic_output,
        order=order,
    )
    editor = RecordingEditor(editor_output, order=order)
    return (
        WritingWorkflow(writer=writer, critic=critic, editor=editor),
        writer,
        critic,
        editor,
    )


def test_pass_calls_writer_and_critic_once_and_never_calls_editor() -> None:
    task = make_task()
    report = pass_report()
    order: list[str] = []
    workflow, writer, critic, editor = make_workflow(
        writer_output="Exact first draft",
        critic_output=report,
        order=order,
    )

    result = workflow.execute(task)

    assert order == ["writer", "critic"]
    assert writer.calls == [task]
    assert critic.calls == [(task, "Exact first draft")]
    assert editor.calls == []
    assert result.writer_output == "Exact first draft"
    assert result.critic_report is report
    assert result.working_draft == result.writer_output
    assert result.revision_applied is False


def test_revise_calls_editor_exactly_once_with_exact_inputs() -> None:
    task = make_task()
    report = revise_report()
    order: list[str] = []
    workflow, writer, critic, editor = make_workflow(
        writer_output="Exact first draft",
        critic_output=report,
        editor_output="Exact updated draft",
        order=order,
    )

    result = workflow.execute(task)

    assert order == ["writer", "critic", "editor"]
    assert writer.calls == [task]
    assert critic.calls == [(task, "Exact first draft")]
    assert editor.calls == [(task, "Exact first draft", report)]
    assert result.writer_output == "Exact first draft"
    assert result.critic_report is report
    assert result.working_draft == "Exact updated draft"
    assert result.revision_applied is True


@pytest.mark.parametrize("output", ["", " ", None, 42])
def test_blank_or_non_text_writer_output_fails_safely(output: object) -> None:
    workflow, _, critic, editor = make_workflow(writer_output=output)

    with pytest.raises(WritingWorkflowError, match=r"^Writer returned invalid output$"):
        workflow.execute(make_task())

    assert critic.calls == []
    assert editor.calls == []


@pytest.mark.parametrize("output", ["", " ", None, 42])
def test_blank_or_non_text_editor_output_fails_safely(output: object) -> None:
    workflow, _, _, editor = make_workflow(
        critic_output=revise_report(),
        editor_output=output,
    )

    with pytest.raises(WritingWorkflowError, match=r"^Editor returned invalid output$"):
        workflow.execute(make_task())

    assert len(editor.calls) == 1


def test_malformed_critic_report_fails_before_editor() -> None:
    malformed = pass_report()
    object.__setattr__(malformed, "verdict", CriticVerdict.REVISE)
    workflow, _, _, editor = make_workflow(critic_output=malformed)

    with pytest.raises(WritingWorkflowError, match=r"^Critic returned an invalid report$"):
        workflow.execute(make_task())

    assert editor.calls == []


@pytest.mark.parametrize(
    ("writer_output", "critic_output", "editor_output", "message"),
    [
        (
            RuntimeError("provider secret writer diagnostics"),
            None,
            "Updated",
            "Writer failed",
        ),
        (
            "Draft",
            RuntimeError("provider secret critic diagnostics"),
            "Updated",
            "Critic failed",
        ),
        (
            "Draft",
            revise_report(),
            RuntimeError("provider secret editor diagnostics"),
            "Editor failed",
        ),
    ],
)
def test_agent_exceptions_are_sanitized(
    writer_output: object,
    critic_output: object,
    editor_output: object,
    message: str,
) -> None:
    workflow, _, _, _ = make_workflow(
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )

    with pytest.raises(WritingWorkflowError, match=rf"^{message}$") as caught:
        workflow.execute(make_task())

    assert "secret" not in str(caught.value).lower()
    assert "diagnostics" not in str(caught.value).lower()
    assert caught.value.__cause__ is None


def test_execute_does_not_mutate_task() -> None:
    task = make_task()
    before = task
    workflow, _, _, _ = make_workflow(critic_output=revise_report())

    workflow.execute(task)

    assert task is before
    assert task == make_task()
    assert task.status is WritingTaskStatus.CREATED
    assert task.working_draft is None
    assert task.critic_report is None


def test_existing_working_draft_reaches_writer_unchanged() -> None:
    task = WritingTask(
        id="task-1",
        conversation_id="conversation-1",
        brief=WritingBrief("Revise the current copy."),
        status=WritingTaskStatus.REVIEWED,
        created_at=NOW,
        updated_at=NOW,
        working_draft="Existing canonical copy",
        critic_report=pass_report(),
    )
    workflow, writer, critic, _ = make_workflow(writer_output="New Writer output")

    result = workflow.execute(task)

    assert writer.calls == [task]
    assert writer.calls[0].working_draft == "Existing canonical copy"
    assert critic.calls == [(task, "New Writer output")]
    assert task.working_draft == "Existing canonical copy"
    assert result.working_draft == "New Writer output"


def test_workflow_instances_do_not_share_mutable_state() -> None:
    first, first_writer, first_critic, first_editor = make_workflow()
    second, second_writer, second_critic, second_editor = make_workflow()

    first.execute(make_task())

    assert len(first_writer.calls) == 1
    assert len(first_critic.calls) == 1
    assert first_editor.calls == []
    assert second_writer.calls == []
    assert second_critic.calls == []
    assert second_editor.calls == []

    second.execute(make_task())
    assert len(first_writer.calls) == len(second_writer.calls) == 1
