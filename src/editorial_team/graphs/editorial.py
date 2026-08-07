"""LangGraph orchestration of one Writer-Critic-Editor attempt."""

from __future__ import annotations

import hashlib
from typing import Literal

from langgraph.graph import END, START, StateGraph

from editorial_team.agents.protocols import Critic, Editor, Writer
from editorial_team.contracts.common import require_non_blank
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
    EditorialResult,
    EditorialRunContext,
)
from editorial_team.errors import ServiceError
from editorial_team.graphs.state import EditorialGraphStateV1, validate_graph_state_version
from editorial_team.tracing import error_category, set_trace_stage, trace_event

EditorialVerdictRoute = Literal["pass", "revise"]


class EditorialGraphError(ServiceError):
    """A sanitized failure from the editorial graph."""


def _route_critic_verdict(state: EditorialGraphStateV1) -> EditorialVerdictRoute:
    report = state.get("critic_report")
    if not isinstance(report, CriticReport):
        raise EditorialGraphError("Critic returned an invalid report")
    return report.verdict.value


class _EditorialNodes:
    def __init__(self, *, writer: Writer, critic: Critic, editor: Editor) -> None:
        self._writer = writer
        self._critic = critic
        self._editor = editor

    def writer(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        validate_graph_state_version(state)
        context = self._context(state)
        if any(
            state.get(field) is not None
            for field in ("writer_output", "critic_report", "editorial_result")
        ):
            raise EditorialGraphError("Editorial run contains stale transient output")
        set_trace_stage("writer")
        trace_event("writer_started", stage="writer", **self._trace_context(context))
        try:
            output = self._writer.write(context)
        except Exception as exc:
            trace_event(
                "writer_failed",
                stage="writer",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise EditorialGraphError("Writer failed") from None
        output = self._text(output, participant="Writer")
        trace_event(
            "writer_completed",
            stage="writer",
            outcome="completed",
            writer_output_hash=self._hash(output),
            **self._trace_context(context),
        )
        return {"writer_output": output, "writer_run_id": context.run_id}

    def critic(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        context = self._context(state)
        self._require_run_id(state, "writer_run_id", context.run_id)
        draft = self._required_text(state, "writer_output")
        set_trace_stage("critic")
        trace_event(
            "critic_started",
            stage="critic",
            writer_output_hash=self._hash(draft),
            **self._trace_context(context),
        )
        try:
            report = self._critic.review(context, draft)
        except Exception as exc:
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise EditorialGraphError("Critic failed") from None
        if not isinstance(report, CriticReport):
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category="schema_validation_failure",
            )
            raise EditorialGraphError("Critic returned an invalid report")
        if (
            context.operation.value != "new_task"
            and context.task.working_draft == draft
            and report.verdict is CriticVerdict.PASS
        ):
            report = CriticReport(
                CriticVerdict.REVISE,
                "The requested transformation was not applied.",
                (
                    CriticIssue(
                        CriticIssueSeverity.MAJOR,
                        "The output is unchanged from the input working draft.",
                        suggestion="Apply the explicit current instruction.",
                        violated_requirement=context.current_instruction,
                        input_evidence="Input and candidate hashes are identical.",
                        candidate_evidence="Candidate is exactly unchanged.",
                    ),
                ),
            )
        if context.operation.value != "new_task" and any(
            issue.violated_requirement is None
            or issue.input_evidence is None
            or issue.candidate_evidence is None
            for issue in report.issues
        ):
            raise EditorialGraphError("Critic returned ungrounded transformation issues")
        try:
            CriticReport(report.verdict, report.summary, report.issues)
        except (TypeError, ValueError):
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category="domain_consistency_failure",
            )
            raise EditorialGraphError("Critic returned an invalid report") from None
        trace_event(
            "critic_completed", stage="critic", outcome="completed", critic_verdict=report.verdict
        )
        return {"critic_report": report, "critic_run_id": context.run_id}

    def editor(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        context = self._context(state)
        self._require_run_id(state, "writer_run_id", context.run_id)
        self._require_run_id(state, "critic_run_id", context.run_id)
        draft = self._required_text(state, "writer_output")
        report = self._report(state)
        set_trace_stage("editor")
        trace_event(
            "editor_started",
            stage="editor",
            writer_output_hash=self._hash(draft),
            **self._trace_context(context),
        )
        try:
            output = self._editor.revise(context, draft, report)
        except Exception as exc:
            trace_event(
                "editor_failed",
                stage="editor",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise EditorialGraphError("Editor failed") from None
        output = self._text(output, participant="Editor")
        trace_event("editor_completed", stage="editor", outcome="completed")
        return {"working_draft": output}

    def build_pass_result(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        context = self._context(state)
        self._require_run_id(state, "writer_run_id", context.run_id)
        self._require_run_id(state, "critic_run_id", context.run_id)
        writer_output = self._required_text(state, "writer_output")
        report = self._report(state)
        return {
            "working_draft": writer_output,
            "editorial_result": self._result(writer_output, report, writer_output, False),
        }

    def build_revised_result(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        context = self._context(state)
        self._require_run_id(state, "writer_run_id", context.run_id)
        self._require_run_id(state, "critic_run_id", context.run_id)
        writer_output = self._required_text(state, "writer_output")
        report = self._report(state)
        working_draft = self._required_text(state, "working_draft")
        return {"editorial_result": self._result(writer_output, report, working_draft, True)}

    @staticmethod
    def _context(state: EditorialGraphStateV1) -> EditorialRunContext:
        context = state.get("editorial_run_context")
        task = state.get("writing_task")
        if not isinstance(context, EditorialRunContext) or task != context.task:
            raise EditorialGraphError("Editorial run context is inconsistent")
        return context

    @staticmethod
    def _require_run_id(
        state: EditorialGraphStateV1,
        field: Literal["writer_run_id", "critic_run_id"],
        expected: str,
    ) -> None:
        if state.get(field) != expected:
            raise EditorialGraphError("Editorial output belongs to another run")

    @staticmethod
    def _report(state: EditorialGraphStateV1) -> CriticReport:
        report = state.get("critic_report")
        if not isinstance(report, CriticReport):
            raise EditorialGraphError("Critic returned an invalid report")
        return report

    @staticmethod
    def _text(value: object, *, participant: str) -> str:
        try:
            return require_non_blank(value, "output")  # type: ignore[arg-type]
        except ValueError:
            trace_event(
                f"{participant.lower()}_failed",
                stage=participant.lower(),
                outcome="failed",
                error_category="blank_response",
            )
            raise EditorialGraphError(f"{participant} returned invalid output") from None

    @classmethod
    def _required_text(cls, state: EditorialGraphStateV1, field: str) -> str:
        return cls._text(state.get(field), participant="Writing workflow")  # type: ignore[literal-required]

    @staticmethod
    def _result(
        writer_output: str, report: CriticReport, working_draft: str, revised: bool
    ) -> EditorialResult:
        try:
            return EditorialResult(writer_output, report, working_draft, revised)
        except (TypeError, ValueError):
            raise EditorialGraphError("Writing workflow produced an invalid result") from None

    @staticmethod
    def _hash(value: str | None) -> str | None:
        return None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _trace_context(cls, context: EditorialRunContext) -> dict[str, object]:
        return {
            "run_id": context.run_id,
            "turn_id": context.turn_id,
            "task_id": context.task.id,
            "operation": context.operation,
            "retrieved_artifact_id": context.retrieved_artifact_id,
            "input_working_draft_hash": cls._hash(context.task.working_draft),
        }


def build_editorial_subgraph(
    *, writer: Writer, critic: Critic, editor: Editor
) -> StateGraph[EditorialGraphStateV1]:
    """Build the sole production Writer-Critic-optional Editor workflow."""

    nodes = _EditorialNodes(writer=writer, critic=critic, editor=editor)
    graph = StateGraph(EditorialGraphStateV1)
    graph.add_node("writer", nodes.writer)
    graph.add_node("critic", nodes.critic)
    graph.add_node("editor", nodes.editor)
    graph.add_node("build_pass_result", nodes.build_pass_result)
    graph.add_node("build_revised_result", nodes.build_revised_result)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic", _route_critic_verdict, {"pass": "build_pass_result", "revise": "editor"}
    )
    graph.add_edge("editor", "build_revised_result")
    graph.add_edge("build_pass_result", END)
    graph.add_edge("build_revised_result", END)
    return graph
