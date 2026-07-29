"""Strict parsing and shared execution for model-backed agents."""

from __future__ import annotations

import json
from typing import Any

from editorial_team.agents.errors import AgentError
from editorial_team.domain.editorial import (
    CriticIssue,
    CriticIssueSeverity,
    CriticReport,
    CriticVerdict,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.models import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    StructuredOutputSpec,
)
from editorial_team.operations.models import (
    AdminAssessment,
    AdminDecision,
    AdminReasonCode,
)


def execute_text(
    model: ModelClient,
    prompt: str,
    role: str,
    *,
    structured_output: StructuredOutputSpec | None = None,
) -> str:
    """Execute one bounded model request and return plain nonblank text."""

    try:
        response = model.respond(
            ModelRequest(input=prompt, structured_output=structured_output)
        )
    except Exception:
        raise AgentError(f"{role} model call failed") from None
    if (
        not isinstance(response, ModelResponse)
        or response.tool_calls
        or not isinstance(response.text, str)
        or not response.text.strip()
    ):
        raise AgentError(f"{role} returned invalid output")
    return response.text


def parse_coordinator_decision(text: str) -> CoordinatorDecision:
    """Parse one strict CoordinatorDecision JSON object."""

    value = _json_object(text)
    _check_keys(
        value,
        required={"route", "confidence"},
        optional={"task_input", "revision_instructions"},
    )
    try:
        return CoordinatorDecision(
            route=CoordinatorRoute(value["route"]),
            confidence=value["confidence"],
            task_input=value.get("task_input"),
            revision_instructions=value.get("revision_instructions"),
        )
    except (KeyError, TypeError, ValueError):
        raise AgentError("Coordinator returned invalid structured output") from None


def parse_critic_report(text: str, draft: str) -> CriticReport:
    """Parse and ground one strict CriticReport JSON object."""

    value = _json_object(text)
    _check_keys(value, required={"verdict", "summary", "issues"}, optional=set())
    issues_value = value["issues"]
    if not isinstance(issues_value, list):
        raise AgentError("Critic returned invalid structured output")

    issues: list[CriticIssue] = []
    for item in issues_value:
        if not isinstance(item, dict):
            raise AgentError("Critic returned invalid structured output")
        _check_keys(
            item,
            required={"severity", "problem"},
            optional={"location", "suggestion", "grounded_excerpt"},
        )
        try:
            issue = CriticIssue(
                severity=CriticIssueSeverity(item["severity"]),
                problem=item["problem"],
                location=item.get("location"),
                suggestion=item.get("suggestion"),
                grounded_excerpt=item.get("grounded_excerpt"),
            )
        except (KeyError, TypeError, ValueError):
            raise AgentError("Critic returned invalid structured output") from None
        if issue.grounded_excerpt is not None and issue.grounded_excerpt not in draft:
            raise AgentError("Critic returned an ungrounded excerpt")
        issues.append(issue)

    try:
        return CriticReport(
            verdict=CriticVerdict(value["verdict"]),
            summary=value["summary"],
            issues=tuple(issues),
        )
    except (KeyError, TypeError, ValueError):
        raise AgentError("Critic returned invalid structured output") from None


def parse_admin_assessment(text: str) -> AdminAssessment:
    """Parse one exact AdminAssessment JSON object."""

    value = _json_object(text)
    _check_keys(value, required={"decision", "reason_code"}, optional=set())
    try:
        return AdminAssessment(
            decision=AdminDecision(value["decision"]),
            reason_code=AdminReasonCode(value["reason_code"]),
        )
    except (KeyError, TypeError, ValueError):
        raise AgentError("Admin returned invalid structured output") from None


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        raise AgentError("Model returned invalid JSON") from None
    if not isinstance(value, dict):
        raise AgentError("Model returned invalid JSON object")
    return value


def _check_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        raise AgentError("Model returned unexpected structured-output fields")
