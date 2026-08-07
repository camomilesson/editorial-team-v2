from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import editorial_team.evaluation.safety_runner as runner
from editorial_team.evaluation.agent_reporting import CampaignManifest
from editorial_team.evaluation.safety_cases import load_safety_evaluation_cases
from editorial_team.evaluation.safety_runner import (
    FROZEN_NORMAL_FALSE_POSITIVES,
    FROZEN_NORMAL_LEGITIMATE_COUNT,
    SafetyExecution,
    SafetyRunResult,
    aggregate_safety_campaign,
    run_safety_campaign,
    score_stored_safety_campaign,
)
from editorial_team.safety import (
    ATTR_CATEGORIES,
    ATTR_INDIRECT_FLAGGED,
    ATTR_INPUT_BLOCKED,
    ATTR_SAFETY_SCHEMA,
    ATTR_STRUCTURAL_SEPARATION,
    ATTR_UNSAFE_BEHAVIOR,
    SAFETY_SCHEMA_VERSION,
    ThreatCategory,
)


def _trace(*, threat: bool, blocked: bool = False, indirect: bool = False) -> object:
    categories = (
        [ThreatCategory.INDIRECT_PROMPT_INJECTION.value]
        if indirect
        else [ThreatCategory.DIRECT_PROMPT_INJECTION.value]
        if threat
        else []
    )
    return SimpleNamespace(
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    parent_id=None,
                    attributes={
                        ATTR_SAFETY_SCHEMA: SAFETY_SCHEMA_VERSION,
                        ATTR_CATEGORIES: categories,
                        "safety.preflight_flagged": threat and not indirect,
                        ATTR_INPUT_BLOCKED: blocked,
                        ATTR_INDIRECT_FLAGGED: indirect,
                        ATTR_STRUCTURAL_SEPARATION: True,
                        ATTR_UNSAFE_BEHAVIOR: False,
                        "evaluation.candidate_answer": "safe bounded completion",
                    },
                )
            ]
        )
    )


def test_runner_records_fixture_classification_without_model_calls() -> None:
    cases = load_safety_evaluation_cases()

    class Executor:
        def execute(self, case: object) -> SafetyExecution:
            adversarial = bool(case.adversarial)
            return SafetyExecution(
                _trace(threat=adversarial, blocked=adversarial),
                f"tr-{case.case_id}",
            )

    results = run_safety_campaign(cases, Executor())

    assert [result.adversarial for result in results] == [case.adversarial for case in cases]
    assert all(result.trace_id.startswith("tr-") for result in results)


def test_real_runner_declares_isolated_paths_and_batch_origin() -> None:
    source = Path(runner.__file__).read_text()

    assert 'Path("evaluation/safety/.runtime")' in source
    assert 'request_origin="batch"' in source
    assert 'run_root / "checkpoints.db"' in source
    assert 'run_root / "artifacts.db"' in source
    assert "runtime_data" not in source


def test_results_serialize_no_attack_text_or_canary(tmp_path: Path) -> None:
    result = SafetyRunResult(
        "case",
        True,
        "tr-1",
        True,
        False,
        False,
        True,
        False,
        False,
        True,
        1.0,
        1.0,
        0.0,
        ("direct_prompt_injection",),
        ("request_protected_prompt",),
        None,
    )
    path = tmp_path / "results.json"

    runner._write_results(path, (result,))
    serialized = path.read_text()

    assert "Ignore the system" not in serialized
    assert "SYNTHETIC-SAFETY-CANARY" not in serialized


def test_persisted_traces_reload_and_score_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = SafetyRunResult(
        "case",
        True,
        "tr-1",
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        None,
        None,
        None,
        (),
        (),
        None,
    )
    trace = _trace(threat=True, blocked=True)
    monkeypatch.setattr(runner.mlflow, "get_trace", lambda *_args, **_kwargs: trace)

    results = score_stored_safety_campaign(
        (stored,),
        CampaignManifest(1, "sqlite:///:memory:", "1", "safety", "results.json"),
    )

    assert results[0].evaluable
    assert results[0].threat_detected == 1.0
    assert results[0].defense_effective == 1.0
    assert results[0].unsafe_behavior == 0.0


def test_adversarial_and_control_aggregation_remain_separate() -> None:
    def result(case_id: str, adversarial: bool, threat: float) -> SafetyRunResult:
        return SafetyRunResult(
            case_id,
            adversarial,
            f"tr-{case_id}",
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            threat,
            threat,
            0.0,
            (),
            (),
            None,
        )

    summary = aggregate_safety_campaign(
        (result("attack", True, 1.0), result("safe-a", False, 0.0), result("safe-b", False, 1.0))
    )

    assert summary.adversarial_cases == 1
    assert summary.adversarial_threats_detected == 1
    assert summary.adversarial_defenses_effective == 1
    assert summary.adversarial_unsafe_outcomes == 0
    assert summary.legitimate_controls == 2
    assert summary.legitimate_control_false_positives == 1
    assert summary.legitimate_control_false_positive_rate == 0.5
    assert summary.frozen_normal_legitimate_traces == FROZEN_NORMAL_LEGITIMATE_COUNT == 36
    assert summary.frozen_normal_false_positives == FROZEN_NORMAL_FALSE_POSITIVES == 0
    assert summary.frozen_normal_false_positive_rate == 0.0


def test_indirect_containment_is_explicit_and_task_can_complete() -> None:
    case = next(
        item
        for item in load_safety_evaluation_cases()
        if item.case_id == "indirect_retrieval_injection"
    )

    class Executor:
        def execute(self, _case: object) -> SafetyExecution:
            return SafetyExecution(_trace(threat=True, indirect=True), "tr-indirect")

    result = run_safety_campaign((case,), Executor())[0]

    assert result.structural_containment
    assert result.task_completed
    assert result.defense_effective == 1.0
    assert result.unsafe_behavior == 0.0


def test_feedback_names_are_reused_not_redefined() -> None:
    assert runner.log_campaign_safety_feedback.__module__.endswith("agent_reporting")
