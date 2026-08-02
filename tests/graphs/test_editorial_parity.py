"""Parity tests for the alternative editorial subgraph."""

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
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.graphs import build_editorial_subgraph
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
    return CriticReport(
        CriticVerdict.REVISE,
        "One revision is required.",
        (
            CriticIssue(
                CriticIssueSeverity.MAJOR,
                "The benefit is unclear.",
                suggestion="State the benefit directly.",
            ),
        ),
    )


@dataclass
class RecordingWriter:
    output: object
    calls: list[WritingTask] = field(default_factory=list)

    def write(self, task: WritingTask) -> Any:
        self.calls.append(task)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingCritic:
    output: object
    calls: list[tuple[WritingTask, str]] = field(default_factory=list)

    def review(self, task: WritingTask, draft: str) -> Any:
        self.calls.append((task, draft))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class RecordingEditor:
    output: object
    calls: list[tuple[WritingTask, str, CriticReport]] = field(default_factory=list)

    def revise(self, task: WritingTask, draft: str, report: CriticReport) -> Any:
        self.calls.append((task, draft, report))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass(frozen=True)
class Participants:
    writer: RecordingWriter
    critic: RecordingCritic
    editor: RecordingEditor


def participants(
    *,
    writer_output: object = "Exact first draft",
    critic_output: object | None = None,
    editor_output: object = "Exact revised draft",
) -> Participants:
    return Participants(
        writer=RecordingWriter(writer_output),
        critic=RecordingCritic(
            pass_report() if critic_output is None else critic_output
        ),
        editor=RecordingEditor(editor_output),
    )


def execute_workflow(task: WritingTask, actors: Participants) -> EditorialResult:
    return WritingWorkflow(
        writer=actors.writer,
        critic=actors.critic,
        editor=actors.editor,
    ).execute(task)


def execute_graph(task: WritingTask, actors: Participants) -> EditorialResult:
    graph = build_editorial_subgraph(
        writer=actors.writer,
        critic=actors.critic,
        editor=actors.editor,
    ).compile()
    state = graph.invoke(
        {
            "state_version": 1,
            "invocation_kind": "external_brief",
            "writing_task": task,
        }
    )
    result = state.get("editorial_result")
    assert isinstance(result, EditorialResult)
    return result


def assert_matching_calls(
    workflow_actors: Participants,
    graph_actors: Participants,
) -> None:
    assert graph_actors.writer.calls == workflow_actors.writer.calls
    assert graph_actors.critic.calls == workflow_actors.critic.calls
    assert graph_actors.editor.calls == workflow_actors.editor.calls


@pytest.mark.parametrize(
    ("report", "expected_editor_calls"),
    [(pass_report(), 0), (revise_report(), 1)],
)
def test_graph_matches_workflow_result_and_call_counts(
    report: CriticReport,
    expected_editor_calls: int,
) -> None:
    task = make_task()
    workflow_actors = participants(critic_output=report)
    graph_actors = participants(critic_output=report)

    workflow_result = execute_workflow(task, workflow_actors)
    graph_result = execute_graph(task, graph_actors)

    assert graph_result == workflow_result
    assert graph_result.critic_report is report
    assert len(graph_actors.writer.calls) == 1
    assert len(graph_actors.critic.calls) == 1
    assert len(graph_actors.editor.calls) == expected_editor_calls
    assert_matching_calls(workflow_actors, graph_actors)


@pytest.mark.parametrize(
    ("writer_output", "critic_output", "editor_output", "message"),
    [
        (RuntimeError("private writer details"), pass_report(), "Revision", "Writer failed"),
        ("Draft", RuntimeError("private critic details"), "Revision", "Critic failed"),
        ("Draft", revise_report(), RuntimeError("private editor details"), "Editor failed"),
    ],
)
def test_graph_matches_workflow_model_failure_behavior(
    writer_output: object,
    critic_output: object,
    editor_output: object,
    message: str,
) -> None:
    task = make_task()
    workflow_actors = participants(
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )
    graph_actors = participants(
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )

    with pytest.raises(WritingWorkflowError) as workflow_error:
        execute_workflow(task, workflow_actors)
    with pytest.raises(WritingWorkflowError) as graph_error:
        execute_graph(task, graph_actors)

    assert str(graph_error.value) == str(workflow_error.value) == message
    assert graph_error.value.__cause__ is workflow_error.value.__cause__ is None
    assert "private" not in str(graph_error.value).lower()
    assert_matching_calls(workflow_actors, graph_actors)


def test_graph_matches_workflow_invalid_critic_output_behavior() -> None:
    malformed = pass_report()
    object.__setattr__(malformed, "verdict", CriticVerdict.REVISE)
    task = make_task()
    workflow_actors = participants(critic_output=malformed)
    graph_actors = participants(critic_output=malformed)

    with pytest.raises(WritingWorkflowError) as workflow_error:
        execute_workflow(task, workflow_actors)
    with pytest.raises(WritingWorkflowError) as graph_error:
        execute_graph(task, graph_actors)

    assert str(graph_error.value) == str(workflow_error.value)
    assert str(graph_error.value) == "Critic returned an invalid report"
    assert graph_actors.editor.calls == workflow_actors.editor.calls == []
    assert_matching_calls(workflow_actors, graph_actors)


@pytest.mark.parametrize(
    ("writer_output", "critic_output", "editor_output", "message"),
    [
        (" ", pass_report(), "Revision", "Writer returned invalid output"),
        ("Draft", revise_report(), None, "Editor returned invalid output"),
    ],
)
def test_graph_matches_workflow_invalid_text_output_behavior(
    writer_output: object,
    critic_output: object,
    editor_output: object,
    message: str,
) -> None:
    task = make_task()
    workflow_actors = participants(
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )
    graph_actors = participants(
        writer_output=writer_output,
        critic_output=critic_output,
        editor_output=editor_output,
    )

    with pytest.raises(WritingWorkflowError) as workflow_error:
        execute_workflow(task, workflow_actors)
    with pytest.raises(WritingWorkflowError) as graph_error:
        execute_graph(task, graph_actors)

    assert str(graph_error.value) == str(workflow_error.value) == message
    assert_matching_calls(workflow_actors, graph_actors)
