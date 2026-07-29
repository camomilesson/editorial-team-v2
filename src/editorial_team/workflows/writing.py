"""Deterministic orchestration of one writing attempt."""

from __future__ import annotations

from editorial_team.agents.protocols import Critic, Editor, Writer
from editorial_team.contracts.common import require_non_blank
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingTask,
)
from editorial_team.errors import ServiceError
from editorial_team.tracing import error_category, set_trace_stage, trace_event


class WritingWorkflowError(ServiceError):
    """A sanitized failure at the writing workflow boundary."""


class WritingWorkflow:
    """Run one Writer, one Critic, and at most one Editor call."""

    def __init__(self, *, writer: Writer, critic: Critic, editor: Editor) -> None:
        self._writer = writer
        self._critic = critic
        self._editor = editor

    def execute(self, task: WritingTask) -> EditorialResult:
        """Produce one internally consistent editorial result."""

        if not isinstance(task, WritingTask):
            raise WritingWorkflowError("Invalid writing task")

        writer_output = self._write(task)
        report = self._review(task, writer_output)

        if report.verdict is CriticVerdict.PASS:
            return self._build_result(
                writer_output=writer_output,
                report=report,
                working_draft=writer_output,
                revision_applied=False,
            )

        working_draft = self._revise(task, writer_output, report)
        return self._build_result(
            writer_output=writer_output,
            report=report,
            working_draft=working_draft,
            revision_applied=True,
        )

    def _write(self, task: WritingTask) -> str:
        set_trace_stage("writer")
        trace_event("writer_started", stage="writer")
        try:
            draft = self._writer.write(task)
        except Exception as exc:
            trace_event(
                "writer_failed",
                stage="writer",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise WritingWorkflowError("Writer failed") from None

        try:
            self._validate_text_output(draft, participant="Writer")
        except WritingWorkflowError:
            trace_event(
                "writer_failed",
                stage="writer",
                outcome="failed",
                error_category="blank_response",
            )
            raise
        trace_event("writer_completed", stage="writer", outcome="completed")
        return draft

    def _review(self, task: WritingTask, draft: str) -> CriticReport:
        set_trace_stage("critic")
        trace_event("critic_started", stage="critic")
        try:
            report = self._critic.review(task, draft)
        except Exception as exc:
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise WritingWorkflowError("Critic failed") from None

        if not isinstance(report, CriticReport):
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category="schema_validation_failure",
            )
            raise WritingWorkflowError("Critic returned an invalid report")

        try:
            CriticReport(
                verdict=report.verdict,
                summary=report.summary,
                issues=report.issues,
            )
        except (TypeError, ValueError):
            trace_event(
                "critic_failed",
                stage="critic",
                outcome="failed",
                error_category="domain_consistency_failure",
            )
            raise WritingWorkflowError("Critic returned an invalid report") from None

        trace_event(
            "critic_completed",
            stage="critic",
            outcome="completed",
            critic_verdict=report.verdict,
        )
        return report

    def _revise(
        self,
        task: WritingTask,
        draft: str,
        report: CriticReport,
    ) -> str:
        set_trace_stage("editor")
        trace_event("editor_started", stage="editor")
        try:
            working_draft = self._editor.revise(task, draft, report)
        except Exception as exc:
            trace_event(
                "editor_failed",
                stage="editor",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise WritingWorkflowError("Editor failed") from None

        try:
            self._validate_text_output(working_draft, participant="Editor")
        except WritingWorkflowError:
            trace_event(
                "editor_failed",
                stage="editor",
                outcome="failed",
                error_category="blank_response",
            )
            raise
        trace_event("editor_completed", stage="editor", outcome="completed")
        return working_draft

    @staticmethod
    def _validate_text_output(value: object, *, participant: str) -> None:
        try:
            require_non_blank(value, "output")  # type: ignore[arg-type]
        except ValueError:
            raise WritingWorkflowError(f"{participant} returned invalid output") from None

    @staticmethod
    def _build_result(
        *,
        writer_output: str,
        report: CriticReport,
        working_draft: str,
        revision_applied: bool,
    ) -> EditorialResult:
        try:
            return EditorialResult(
                writer_output=writer_output,
                critic_report=report,
                working_draft=working_draft,
                revision_applied=revision_applied,
            )
        except (TypeError, ValueError):
            raise WritingWorkflowError("Writing workflow produced an invalid result") from None
