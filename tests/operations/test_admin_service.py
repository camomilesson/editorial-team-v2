"""Tests for deterministic Admin validation and heartbeat persistence."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from editorial_team.agents import AgentError
from editorial_team.operations import (
    AdminAssessment,
    AdminDecision,
    AdminPolicy,
    AdminPolicyMismatchError,
    AdminReasonCode,
    HeartbeatEvaluationError,
    HeartbeatEvaluationService,
    HeartbeatResult,
    OperationalSnapshot,
    SQLiteHeartbeatResultStore,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def snapshot(**changes: object) -> OperationalSnapshot:
    values = {
        "observed_at": NOW,
        "worker_running": True,
        "queue_depth": 0,
        "queue_capacity": 100,
        "completed_jobs": 1,
        "failed_jobs": 0,
        "last_success_at": NOW,
    }
    values.update(changes)
    return OperationalSnapshot(**values)  # type: ignore[arg-type]


@dataclass
class RecordingAdmin:
    output: AdminAssessment | Exception

    def __post_init__(self) -> None:
        self.calls: list[tuple[OperationalSnapshot, AdminPolicy]] = []

    def evaluate(
        self,
        runtime_snapshot: OperationalSnapshot,
        policy: AdminPolicy,
    ) -> AdminAssessment:
        self.calls.append((runtime_snapshot, policy))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class RecordingStore:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.saved: list[HeartbeatResult] = []
        self.mark_calls: list[str] = []

    def save(self, result: HeartbeatResult) -> None:
        if self.failure is not None:
            raise self.failure
        self.saved.append(result)

    def mark_notification_sent(self, result_id: str) -> HeartbeatResult:
        self.mark_calls.append(result_id)
        raise AssertionError("must not mark notifications")


@pytest.mark.parametrize(
    ("runtime_snapshot", "assessment"),
    [
        (
            snapshot(),
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            ),
        ),
        (
            snapshot(worker_running=False),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.WORKER_STOPPED,
            ),
        ),
        (
            snapshot(failed_jobs=3),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.REPEATED_FAILURES,
            ),
        ),
        (
            snapshot(queue_depth=80),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.QUEUE_PRESSURE,
            ),
        ),
    ],
)
def test_valid_assessment_is_preserved_and_saved_once(
    runtime_snapshot: OperationalSnapshot,
    assessment: AdminAssessment,
) -> None:
    admin = RecordingAdmin(assessment)
    store = RecordingStore()
    service = HeartbeatEvaluationService(
        admin_agent=admin,
        store=store,  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "heartbeat-generated",
    )

    result = service.evaluate_and_store(runtime_snapshot, correlation_id="heartbeat-run-1")

    assert result == store.saved[0]
    assert result.id == "heartbeat-generated"
    assert result.snapshot is runtime_snapshot
    assert result.decision is assessment.decision
    assert result.reason_code is assessment.reason_code
    assert result.notification_sent is False
    assert admin.calls == [(runtime_snapshot, AdminPolicy())]
    assert len(store.saved) == 1
    assert store.mark_calls == []


@pytest.mark.parametrize(
    ("runtime_snapshot", "wrong_assessment"),
    [
        (
            snapshot(worker_running=False),
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            ),
        ),
        (
            snapshot(worker_running=False, failed_jobs=3, queue_depth=90),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.REPEATED_FAILURES,
            ),
        ),
        (
            snapshot(failed_jobs=3, queue_depth=90),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.QUEUE_PRESSURE,
            ),
        ),
    ],
)
def test_policy_mismatch_saves_nothing(
    runtime_snapshot: OperationalSnapshot,
    wrong_assessment: AdminAssessment,
) -> None:
    store = RecordingStore()
    service = HeartbeatEvaluationService(
        admin_agent=RecordingAdmin(wrong_assessment),
        store=store,  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "unused",
    )

    with pytest.raises(AdminPolicyMismatchError):
        service.evaluate_and_store(runtime_snapshot, correlation_id="heartbeat-run-1")

    assert store.saved == []


def test_agent_failure_is_sanitized_and_saves_nothing() -> None:
    admin = RecordingAdmin(AgentError("RAW-AGENT-SECRET"))
    store = RecordingStore()
    service = HeartbeatEvaluationService(
        admin_agent=admin,
        store=store,  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "unused",
    )

    with pytest.raises(HeartbeatEvaluationError) as error:
        service.evaluate_and_store(snapshot(), correlation_id="heartbeat-run-1")

    assert str(error.value) == "Heartbeat evaluation failed"
    assert store.saved == []
    assert len(admin.calls) == 1


def test_repository_failure_is_sanitized() -> None:
    service = HeartbeatEvaluationService(
        admin_agent=RecordingAdmin(
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            )
        ),
        store=RecordingStore(RuntimeError("DATABASE-PATH-AND-SQL-SECRET")),  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "heartbeat-1",
    )

    with pytest.raises(HeartbeatEvaluationError) as error:
        service.evaluate_and_store(snapshot(), correlation_id="heartbeat-run-1")

    assert str(error.value) == "Heartbeat evaluation failed"


def test_service_rejects_non_operational_input() -> None:
    service = HeartbeatEvaluationService(
        admin_agent=RecordingAdmin(
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            )
        ),
        store=RecordingStore(),  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "heartbeat-1",
    )

    with pytest.raises(ValueError, match="OperationalSnapshot"):
        service.evaluate_and_store(object(), correlation_id="heartbeat-run-1")  # type: ignore[arg-type]


def test_real_sqlite_integration_persists_silence_and_notify(tmp_path: Path) -> None:
    database = tmp_path / "heartbeats.db"
    store = SQLiteHeartbeatResultStore(database)
    store.initialize()
    outputs = iter(
        [
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            ),
            AdminAssessment(
                AdminDecision.NOTIFY,
                AdminReasonCode.WORKER_STOPPED,
            ),
        ]
    )

    class SequentialAdmin:
        def evaluate(
            self,
            runtime_snapshot: OperationalSnapshot,
            policy: AdminPolicy,
        ) -> AdminAssessment:
            return next(outputs)

    ids = iter(["heartbeat-healthy", "heartbeat-stopped"])
    service = HeartbeatEvaluationService(
        admin_agent=SequentialAdmin(),
        store=store,
        policy=AdminPolicy(),
        identifier_generator=lambda: next(ids),
    )
    service.evaluate_and_store(snapshot(), correlation_id="run-healthy")
    service.evaluate_and_store(
        snapshot(worker_running=False),
        correlation_id="run-stopped",
    )

    reopened = SQLiteHeartbeatResultStore(database)
    healthy = reopened.get("heartbeat-healthy")
    stopped = reopened.get("heartbeat-stopped")
    assert healthy.notification_sent is False
    assert stopped.notification_sent is False
    assert healthy.decision is AdminDecision.SILENCE
    assert stopped.decision is AdminDecision.NOTIFY


def trace_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "editorial_team.live_trace"
    ]


def test_success_tracing_contains_only_safe_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    service = HeartbeatEvaluationService(
        admin_agent=RecordingAdmin(
            AdminAssessment(
                AdminDecision.SILENCE,
                AdminReasonCode.SYSTEM_HEALTHY,
            )
        ),
        store=RecordingStore(),  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "heartbeat-1",
    )

    service.evaluate_and_store(snapshot(), correlation_id="heartbeat-run-1")

    trace = trace_messages(caplog)
    assert [message.split()[0] for message in trace] == [
        "admin_started",
        "admin_completed",
        "heartbeat_result_saved",
    ]
    assert "decision=silence" in trace[1]
    assert "reason_code=system_healthy" in trace[1]
    assert "result_id=heartbeat-1" in trace[2]


@pytest.mark.parametrize(
    ("admin_output", "store_failure", "category"),
    [
        (
            AdminAssessment(AdminDecision.SILENCE, AdminReasonCode.SYSTEM_HEALTHY),
            RuntimeError("DATABASE-PATH-SECRET"),
            "runtime_error",
        ),
        (
            AgentError("Admin model call failed"),
            None,
            "provider_model_failure",
        ),
        (
            AdminAssessment(AdminDecision.NOTIFY, AdminReasonCode.WORKER_STOPPED),
            None,
            "admin_policy_mismatch_error",
        ),
    ],
)
def test_failure_tracing_is_sanitized_and_never_saved(
    caplog: pytest.LogCaptureFixture,
    admin_output: AdminAssessment | Exception,
    store_failure: Exception | None,
    category: str,
) -> None:
    caplog.set_level(logging.INFO, logger="editorial_team.live_trace")
    service = HeartbeatEvaluationService(
        admin_agent=RecordingAdmin(admin_output),
        store=RecordingStore(store_failure),  # type: ignore[arg-type]
        policy=AdminPolicy(),
        identifier_generator=lambda: "heartbeat-1",
    )

    with pytest.raises(HeartbeatEvaluationError):
        service.evaluate_and_store(snapshot(), correlation_id="heartbeat-run-1")

    trace = "\n".join(trace_messages(caplog))
    assert "admin_started" in trace
    assert "admin_failed" in trace
    assert f"error_category={category}" in trace
    assert "heartbeat_result_saved" not in trace
    for secret in (
        "DATABASE-PATH-SECRET",
        "PROMPT-SECRET",
        "RAW-OUTPUT-SECRET",
        "USER-TEXT-SECRET",
        "DRAFT-TEXT-SECRET",
    ):
        assert secret not in trace
