"""Stable transport-neutral formatting for visible agent messages."""

from __future__ import annotations

from editorial_team.domain.editorial import CriticReport, EditorialResult


def format_agent_message(agent: str, content: str) -> str:
    """Render one visible agent heading and body."""

    return f"{agent}\n\n{content}"


def format_talker_message(content: str) -> str:
    """Render one Talker response."""

    return format_agent_message("💬 Talker", content)


def format_writer_message(content: str) -> str:
    """Render exact Writer output."""

    return format_agent_message("✍️ Writer", content)


def format_critic_report(report: CriticReport) -> str:
    """Render a Critic report with readable, ordered issue fields."""

    sections = [
        f"Verdict: {report.verdict.value.upper()}",
        f"Summary: {report.summary}",
    ]
    if not report.issues:
        sections.append("Issues: None")
        return format_agent_message("🔍 Critic", "\n\n".join(sections))

    issue_sections: list[str] = []
    for index, issue in enumerate(report.issues, start=1):
        fields = [f"{index}. Severity: {issue.severity.value.upper()}"]
        if issue.location is not None:
            fields.append(f"Location: {issue.location}")
        fields.append(f"Problem: {issue.problem}")
        if issue.suggestion is not None:
            fields.append(f"Suggestion: {issue.suggestion}")
        if issue.grounded_excerpt is not None:
            fields.append(f"Grounded excerpt: {issue.grounded_excerpt}")
        issue_sections.append("\n\n".join(fields))
    sections.append("Issues:\n\n" + "\n\n".join(issue_sections))
    return format_agent_message("Critic", "\n\n".join(sections))


def format_editor_message(result: EditorialResult) -> str:
    """Render revised output or the deterministic PASS handoff."""

    content = (
        result.working_draft if result.revision_applied else "Working draft approved, see above."
    )
    return format_agent_message("🛠️ Editor", content)
