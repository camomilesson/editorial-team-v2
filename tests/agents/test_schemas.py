from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from editorial_team.agents.parsing import (
    parse_coordinator_decision,
    parse_critic_report,
)
from editorial_team.agents.schemas import (
    COORDINATOR_DECISION_SCHEMA,
    CRITIC_REPORT_SCHEMA,
)
from editorial_team.domain.editorial import CriticIssueSeverity, CriticVerdict
from editorial_team.domain.routing import CoordinatorRoute


def validate(schema: dict[str, object], payload: dict[str, object]) -> None:
    Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"route": "chat", "confidence": 1},
        {
            "route": "start_writing_task",
            "confidence": 0.9,
            "task_input": "Write a post.",
            "revision_instructions": None,
        },
        {
            "route": "revise_task",
            "confidence": 0.7,
            "task_input": None,
            "revision_instructions": "Make it shorter.",
        },
    ],
)
def test_coordinator_schema_accepts_parser_contract(payload: dict[str, object]) -> None:
    validate(COORDINATOR_DECISION_SCHEMA, payload)

    parse_coordinator_decision(json.dumps(payload))


def test_coordinator_schema_matches_domain_enums_and_optional_keys() -> None:
    properties = COORDINATOR_DECISION_SCHEMA["properties"]

    assert properties["route"]["enum"] == [route.value for route in CoordinatorRoute]
    assert COORDINATOR_DECISION_SCHEMA["required"] == ["route", "confidence"]
    assert properties["task_input"]["type"] == ["string", "null"]
    assert properties["revision_instructions"]["type"] == ["string", "null"]
    assert COORDINATOR_DECISION_SCHEMA["additionalProperties"] is False
    assert properties["route"]["enum"] == [
        "chat",
        "start_writing_task",
        "revise_task",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "pass", "summary": "Ready.", "issues": []},
        {
            "verdict": "revise",
            "summary": "Revise the opening.",
            "issues": [
                {
                    "severity": "major",
                    "location": "Opening",
                    "problem": "The opening is vague.",
                    "suggestion": "Name the subject.",
                    "grounded_excerpt": "Current opening",
                }
            ],
        },
    ],
)
def test_critic_schema_accepts_parser_contract(payload: dict[str, object]) -> None:
    validate(CRITIC_REPORT_SCHEMA, payload)

    parse_critic_report(json.dumps(payload), "Current opening and body.")


def test_critic_schema_matches_domain_enums_required_and_nullable_fields() -> None:
    properties = CRITIC_REPORT_SCHEMA["properties"]
    issue_schema = properties["issues"]["items"]
    issue_properties = issue_schema["properties"]

    assert properties["verdict"]["enum"] == [
        verdict.value for verdict in CriticVerdict
    ]
    assert CRITIC_REPORT_SCHEMA["required"] == ["verdict", "summary", "issues"]
    assert issue_properties["severity"]["enum"] == [
        severity.value for severity in CriticIssueSeverity
    ]
    assert issue_schema["required"] == ["severity", "problem"]
    assert issue_properties["location"]["type"] == ["string", "null"]
    assert issue_properties["suggestion"]["type"] == ["string", "null"]
    assert issue_properties["grounded_excerpt"]["type"] == ["string", "null"]
    assert CRITIC_REPORT_SCHEMA["additionalProperties"] is False
    assert issue_schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (
            COORDINATOR_DECISION_SCHEMA,
            {"route": "chat", "confidence": 1, "unknown": True},
        ),
        (
            CRITIC_REPORT_SCHEMA,
            {"verdict": "pass", "summary": "Ready.", "issues": [], "unknown": True},
        ),
    ],
)
def test_structured_schemas_prohibit_unknown_keys(
    schema: dict[str, object],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        validate(schema, payload)
