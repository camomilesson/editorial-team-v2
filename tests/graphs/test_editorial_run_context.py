"""Focused ownership tests for one immutable editorial subgraph run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    EditorialOperation,
    EditorialRunContext,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.graphs.editorial import EditorialGraphError, build_editorial_subgraph

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def task(task_id: str, *, draft: str = "Source sun draft") -> WritingTask:
    return WritingTask(
        task_id,
        "conversation-1",
        WritingBrief("Write a short post about the sun being nice", ("Make it shorter",)),
        WritingTaskStatus.CREATED,
        NOW,
        NOW,
        draft,
    )


def context(task_id: str, turn_id: str) -> EditorialRunContext:
    return EditorialRunContext(
        turn_id,
        EditorialOperation.HISTORICAL_TRANSFORMATION,
        task(task_id),
        "Make it shorter",
        "sun-artifact",
    )


@dataclass
class WriterSpy:
    contexts: list[EditorialRunContext] = field(default_factory=list)

    def write(self, run: EditorialRunContext) -> str:
        self.contexts.append(run)
        return run.task.working_draft or "missing"


@dataclass
class CriticSpy:
    contexts: list[EditorialRunContext] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)

    def review(self, run: EditorialRunContext, draft: str) -> CriticReport:
        self.contexts.append(run)
        self.drafts.append(draft)
        return CriticReport(CriticVerdict.PASS, "Looks fluent")


@dataclass
class EditorSpy:
    contexts: list[EditorialRunContext] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)
    reports: list[CriticReport] = field(default_factory=list)

    def revise(
        self,
        run: EditorialRunContext,
        draft: str,
        report: CriticReport,
    ) -> str:
        self.contexts.append(run)
        self.drafts.append(draft)
        self.reports.append(report)
        return "Short sun draft"


def runner(
    writer: WriterSpy,
    critic: CriticSpy,
    editor: EditorSpy,
) -> object:
    return build_editorial_subgraph(writer=writer, critic=critic, editor=editor).compile()


def invocation(run: EditorialRunContext) -> dict[str, object]:
    return {
        "state_version": 1,
        "invocation_kind": "conversation",
        "writing_task": run.task,
        "editorial_run_context": run,
        "writer_output": None,
        "critic_report": None,
        "editorial_result": None,
        "writer_run_id": None,
        "critic_run_id": None,
    }


def test_all_agents_share_one_context_and_unchanged_transformation_cannot_pass() -> None:
    writer, critic, editor = WriterSpy(), CriticSpy(), EditorSpy()
    run = context("run-1", "turn-1")

    result = runner(writer, critic, editor).invoke(invocation(run))  # type: ignore[attr-defined]

    assert writer.contexts == critic.contexts == editor.contexts == [run]
    assert critic.drafts == editor.drafts == ["Source sun draft"]
    assert editor.reports[0].verdict is CriticVerdict.REVISE
    assert "unchanged" in editor.reports[0].issues[0].problem
    assert result["writer_run_id"] == result["critic_run_id"] == run.run_id
    assert result["editorial_result"].working_draft == "Short sun draft"


def test_consecutive_runs_have_distinct_contexts_and_no_output_leaks() -> None:
    writer, critic, editor = WriterSpy(), CriticSpy(), EditorSpy()
    graph = runner(writer, critic, editor)
    first = context("run-1", "turn-1")
    second = context("run-2", "turn-2")

    graph.invoke(invocation(first))  # type: ignore[attr-defined]
    graph.invoke(invocation(second))  # type: ignore[attr-defined]

    assert writer.contexts == [first, second]
    assert critic.contexts == [first, second]
    assert editor.contexts == [first, second]
    assert first.run_id != second.run_id


def test_mismatched_task_and_context_are_rejected_before_writer() -> None:
    writer, critic, editor = WriterSpy(), CriticSpy(), EditorSpy()
    run = context("run-1", "turn-1")
    state = invocation(run)
    state["writing_task"] = task("another-run")

    with pytest.raises(EditorialGraphError, match="context is inconsistent"):
        runner(writer, critic, editor).invoke(state)  # type: ignore[attr-defined]

    assert writer.contexts == [] and critic.contexts == [] and editor.contexts == []


def test_stale_prior_turn_output_is_rejected_before_writer() -> None:
    writer, critic, editor = WriterSpy(), CriticSpy(), EditorSpy()
    state = invocation(context("run-1", "turn-1"))
    state["editorial_result"] = object()

    with pytest.raises(EditorialGraphError, match="stale transient output"):
        runner(writer, critic, editor).invoke(state)  # type: ignore[attr-defined]

    assert writer.contexts == []
