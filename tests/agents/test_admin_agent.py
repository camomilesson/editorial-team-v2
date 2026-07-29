"""Tests for strict, capability-restricted LLM AdminAgent."""

import json
from datetime import UTC, datetime

import jsonschema
import pytest

from editorial_team.agents import AgentError, LlmAdminAgent
from editorial_team.agents.parsing import parse_admin_assessment
from editorial_team.agents.prompts import admin_prompt
from editorial_team.agents.schemas import (
    ADMIN_ASSESSMENT_SCHEMA,
    ADMIN_STRUCTURED_OUTPUT,
)
from editorial_team.models import (
    FakeModelClient,
    ModelClientError,
    ModelRequest,
    ModelResponse,
)
from editorial_team.operations import (
    AdminAssessment,
    AdminDecision,
    AdminPolicy,
    AdminReasonCode,
    OperationalSnapshot,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def snapshot() -> OperationalSnapshot:
    return OperationalSnapshot(
        observed_at=NOW,
        worker_running=True,
        queue_depth=8,
        queue_capacity=100,
        completed_jobs=12,
        failed_jobs=1,
        last_success_at=NOW,
    )


def response(value: object) -> ModelResponse:
    text = value if isinstance(value, str) else json.dumps(value)
    return ModelResponse(text=text, tool_calls=(), continuation_token=None)


@pytest.mark.parametrize(
    "value",
    [
        {"decision": "silence", "reason_code": "system_healthy"},
        {"decision": "notify", "reason_code": "worker_stopped"},
        {"decision": "notify", "reason_code": "repeated_failures"},
        {"decision": "notify", "reason_code": "queue_pressure"},
    ],
)
def test_admin_schema_accepts_every_valid_assessment(value: dict[str, str]) -> None:
    jsonschema.validate(value, ADMIN_ASSESSMENT_SCHEMA)


@pytest.mark.parametrize(
    "value",
    [
        {"decision": "silence"},
        {"reason_code": "system_healthy"},
        {"decision": "silence", "reason_code": "system_healthy", "extra": 1},
        {"decision": "other", "reason_code": "system_healthy"},
        {"decision": "silence", "reason_code": "other"},
    ],
)
def test_admin_schema_rejects_invalid_shapes(value: dict[str, object]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, ADMIN_ASSESSMENT_SCHEMA)


def test_schema_enum_values_exactly_match_domain() -> None:
    properties = ADMIN_ASSESSMENT_SCHEMA["properties"]

    assert properties["decision"]["enum"] == [value.value for value in AdminDecision]
    assert properties["reason_code"]["enum"] == [
        value.value for value in AdminReasonCode
    ]
    assert set(ADMIN_ASSESSMENT_SCHEMA) == {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }


def test_valid_assessment_parses() -> None:
    assert parse_admin_assessment(
        '{"decision":"notify","reason_code":"queue_pressure"}'
    ) == AdminAssessment(AdminDecision.NOTIFY, AdminReasonCode.QUEUE_PRESSURE)


@pytest.mark.parametrize(
    "text",
    [
        "not-json",
        "[]",
        '{"decision":"silence"}',
        '{"decision":"silence","reason_code":"system_healthy","extra":true}',
        '{"decision":"invalid","reason_code":"system_healthy"}',
        '{"decision":"notify","reason_code":"system_healthy"}',
        '```json\n{"decision":"silence","reason_code":"system_healthy"}\n```',
        'prefix {"decision":"silence","reason_code":"system_healthy"}',
    ],
)
def test_parser_rejects_malformed_or_inconsistent_output(text: str) -> None:
    with pytest.raises(AgentError):
        parse_admin_assessment(text)


def test_prompt_contains_only_safe_snapshot_and_policy_fields() -> None:
    prompt = admin_prompt(snapshot(), AdminPolicy())

    for expected in (
        '"observed_at"',
        '"worker_running"',
        '"queue_depth"',
        '"queue_capacity"',
        '"completed_jobs"',
        '"failed_jobs"',
        '"last_success_at"',
        '"failure_threshold"',
        '"queue_pressure_ratio"',
        '"priority_order"',
    ):
        assert expected in prompt
    for secret in (
        "CHAT-ID-SECRET",
        "USERNAME-SECRET",
        "USER-MESSAGE-SECRET",
        "DRAFT-SECRET",
        "DATABASE-PATH-SECRET",
    ):
        assert secret not in prompt


def test_llm_admin_uses_exact_structured_output_without_tools() -> None:
    model = FakeModelClient(
        [response({"decision": "silence", "reason_code": "system_healthy"})]
    )

    assessment = LlmAdminAgent(model).evaluate(snapshot(), AdminPolicy())

    assert assessment == AdminAssessment(
        AdminDecision.SILENCE,
        AdminReasonCode.SYSTEM_HEALTHY,
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.structured_output == ADMIN_STRUCTURED_OUTPUT
    assert request.tools == ()
    assert request.continuation_token is None


@pytest.mark.parametrize(
    ("model_response", "message"),
    [
        (response(" "), "Admin returned invalid output"),
        (response("not-json"), "Model returned invalid JSON"),
    ],
)
def test_llm_admin_invalid_response_fails_once(
    model_response: ModelResponse,
    message: str,
) -> None:
    model = FakeModelClient([model_response, response({})])

    with pytest.raises(AgentError, match=message):
        LlmAdminAgent(model).evaluate(snapshot(), AdminPolicy())

    assert len(model.requests) == 1


class FailingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def respond(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise ModelClientError("PROVIDER-DIAGNOSTIC-SECRET")


def test_provider_failure_is_sanitized_without_retry() -> None:
    model = FailingModel()

    with pytest.raises(AgentError) as error:
        LlmAdminAgent(model).evaluate(snapshot(), AdminPolicy())

    assert str(error.value) == "Admin model call failed"
    assert len(model.requests) == 1
