"""Alternative LangGraph orchestration of one editorial attempt."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from editorial_team.agents.protocols import Critic, Editor, Writer
from editorial_team.domain.editorial import CriticReport, CriticVerdict, WritingTask
from editorial_team.graphs.state import (
    EditorialGraphStateV1,
    validate_graph_state_version,
)
from editorial_team.workflows.writing import WritingWorkflow, WritingWorkflowError

EditorialVerdictRoute = Literal["pass", "revise"]


def _route_critic_verdict(state: EditorialGraphStateV1) -> EditorialVerdictRoute:
    """Describe the future Critic verdict branch without invoking an agent."""

    report = state.get("critic_report")
    if report is None:
        raise ValueError("Editorial graph state has no Critic report")
    if report.verdict is CriticVerdict.PASS:
        return "pass"
    return "revise"


class _EditorialNodes:
    """Node adapters around the existing validated workflow steps."""

    def __init__(self, workflow: WritingWorkflow) -> None:
        self._workflow = workflow

    def writer(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Produce and validate exactly one Writer output."""

        validate_graph_state_version(state)
        task = state.get("writing_task")
        if not isinstance(task, WritingTask):
            raise WritingWorkflowError("Invalid writing task")
        return {"writer_output": self._workflow._write(task)}

    def critic(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Review and validate the exact Writer output once."""

        task = self._require_task(state)
        writer_output = self._require_text(state, "writer_output")
        return {"critic_report": self._workflow._review(task, writer_output)}

    def editor(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Revise once using the exact task, draft, and Critic report."""

        task = self._require_task(state)
        writer_output = self._require_text(state, "writer_output")
        report = self._require_report(state)
        return {
            "working_draft": self._workflow._revise(
                task,
                writer_output,
                report,
            )
        }

    def build_pass_result(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Build the same immutable result as the workflow PASS path."""

        writer_output = self._require_text(state, "writer_output")
        report = self._require_report(state)
        return {
            "working_draft": writer_output,
            "editorial_result": self._workflow._build_result(
                writer_output=writer_output,
                report=report,
                working_draft=writer_output,
                revision_applied=False,
            ),
        }

    def build_revised_result(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Build the same immutable result as the workflow REVISE path."""

        writer_output = self._require_text(state, "writer_output")
        report = self._require_report(state)
        working_draft = self._require_text(state, "working_draft")
        return {
            "editorial_result": self._workflow._build_result(
                writer_output=writer_output,
                report=report,
                working_draft=working_draft,
                revision_applied=True,
            )
        }

    @staticmethod
    def _require_task(state: EditorialGraphStateV1) -> WritingTask:
        task = state.get("writing_task")
        if not isinstance(task, WritingTask):
            raise WritingWorkflowError("Invalid writing task")
        return task

    @staticmethod
    def _require_text(state: EditorialGraphStateV1, field_name: str) -> str:
        value = state.get(field_name)  # type: ignore[literal-required]
        if not isinstance(value, str):
            raise WritingWorkflowError("Writing workflow produced an invalid result")
        return value

    @staticmethod
    def _require_report(state: EditorialGraphStateV1) -> CriticReport:
        report = state.get("critic_report")
        if not isinstance(report, CriticReport):
            raise WritingWorkflowError("Critic returned an invalid report")
        return report


def build_editorial_subgraph(
    *,
    writer: Writer,
    critic: Critic,
    editor: Editor,
) -> StateGraph[EditorialGraphStateV1]:
    """Build the disconnected Writer-Critic-Editor alternative.

    The graph reuses the current workflow's validated step implementation but is
    not connected to application composition or any production request path.
    """

    nodes = _EditorialNodes(
        WritingWorkflow(writer=writer, critic=critic, editor=editor)
    )
    graph = StateGraph(EditorialGraphStateV1)
    graph.add_node("writer", nodes.writer)
    graph.add_node("critic", nodes.critic)
    graph.add_node("editor", nodes.editor)
    graph.add_node("build_pass_result", nodes.build_pass_result)
    graph.add_node("build_revised_result", nodes.build_revised_result)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_critic_verdict,
        {
            "pass": "build_pass_result",
            "revise": "editor",
        },
    )
    graph.add_edge("editor", "build_revised_result")
    graph.add_edge("build_pass_result", END)
    graph.add_edge("build_revised_result", END)
    return graph
