"""Stable transport-neutral formatting for writing-cycle messages."""

from __future__ import annotations

from editorial_team.domain.editorial import CriticReport, EditorialResult


def format_critic_report(report: CriticReport) -> str:
    """Render a critic report without implementation-specific representations."""

    lines = [
        f"Critic verdict: {report.verdict.value.upper()}",
        f"Summary: {report.summary}",
    ]
    if report.issues:
        lines.append("Issues:")
        for index, issue in enumerate(report.issues, start=1):
            lines.append(f"{index}. Severity: {issue.severity.value.upper()}")
            if issue.location is not None:
                lines.append(f"   Location: {issue.location}")
            lines.append(f"   Problem: {issue.problem}")
            if issue.suggestion is not None:
                lines.append(f"   Suggestion: {issue.suggestion}")
            if issue.grounded_excerpt is not None:
                lines.append(f"   Grounded excerpt: {issue.grounded_excerpt}")
    else:
        lines.append("Issues: None")
    return "\n".join(lines)


def format_working_draft(result: EditorialResult) -> str:
    """Render the canonical draft outcome of a writing cycle."""

    if result.revision_applied:
        return f"Revised working draft:\n{result.working_draft}"
    return "The Writer output is now the working draft."


def request_user_evaluation() -> str:
    """Return the stable evaluation request shown after a writing cycle."""

    return "Please review the working draft and tell me whether you approve it or want changes."
