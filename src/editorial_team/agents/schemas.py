"""Provider-neutral JSON schemas for structured editorial-agent output."""

from __future__ import annotations

from typing import Any

from editorial_team.domain.editorial import CriticIssueSeverity, CriticVerdict
from editorial_team.domain.routing import CoordinatorRoute
from editorial_team.models import StructuredOutputSpec

JSON_MIME_TYPE = "application/json"

COORDINATOR_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": [route.value for route in CoordinatorRoute]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "task_input": {"type": ["string", "null"], "minLength": 1},
        "revision_instructions": {"type": ["string", "null"], "minLength": 1},
    },
    "required": ["route", "confidence"],
    "additionalProperties": False,
}

CRITIC_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [verdict.value for verdict in CriticVerdict],
        },
        "summary": {"type": "string", "minLength": 1},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": [severity.value for severity in CriticIssueSeverity],
                    },
                    "location": {"type": ["string", "null"], "minLength": 1},
                    "problem": {"type": "string", "minLength": 1},
                    "suggestion": {"type": ["string", "null"], "minLength": 1},
                    "grounded_excerpt": {"type": ["string", "null"], "minLength": 1},
                },
                "required": ["severity", "problem"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdict", "summary", "issues"],
    "additionalProperties": False,
}

COORDINATOR_STRUCTURED_OUTPUT = StructuredOutputSpec(
    JSON_MIME_TYPE,
    COORDINATOR_DECISION_SCHEMA,
)
CRITIC_STRUCTURED_OUTPUT = StructuredOutputSpec(
    JSON_MIME_TYPE,
    CRITIC_REPORT_SCHEMA,
)
