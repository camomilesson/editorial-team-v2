import math

import pytest

from editorial_team.domain.routing import (
    ClarificationReason,
    CoordinatorDecision,
    CoordinatorRoute,
    TalkerContext,
)


def test_coordinator_routes_are_exactly_chat_start_and_revise() -> None:
    assert {route.value for route in CoordinatorRoute} == {
        "chat",
        "start_writing_task",
        "revise_task",
    }


@pytest.mark.parametrize(
    "decision",
    [
        CoordinatorDecision(CoordinatorRoute.CHAT, 0.6),
        CoordinatorDecision(
            CoordinatorRoute.CHAT,
            0.8,
            talker_context=TalkerContext(
                ClarificationReason.NO_MATCH,
                (),
                "Could you share another clue?",
            ),
        ),
        CoordinatorDecision(
            CoordinatorRoute.START_WRITING_TASK,
            1.0,
            task_input="Write a product announcement.",
        ),
        CoordinatorDecision(
            CoordinatorRoute.REVISE_TASK,
            0.75,
            revision_instructions="Use a warmer opening.",
        ),
    ],
)
def test_constructs_valid_coordinator_decisions(decision: CoordinatorDecision) -> None:
    assert 0 <= decision.confidence <= 1


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.nan, math.inf, True])
def test_confidence_must_be_finite_and_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        CoordinatorDecision(CoordinatorRoute.CHAT, confidence)


@pytest.mark.parametrize(
    "decision",
    [
        lambda: CoordinatorDecision(CoordinatorRoute.START_WRITING_TASK, 0.8),
        lambda: CoordinatorDecision(
            CoordinatorRoute.START_WRITING_TASK,
            0.8,
            task_input="Write this.",
            revision_instructions="Change it.",
        ),
        lambda: CoordinatorDecision(CoordinatorRoute.REVISE_TASK, 0.8),
        lambda: CoordinatorDecision(
            CoordinatorRoute.REVISE_TASK,
            0.8,
            task_input="Write this.",
            revision_instructions="Change it.",
        ),
        lambda: CoordinatorDecision(CoordinatorRoute.CHAT, 0.8, task_input="Write this."),
    ],
)
def test_route_payload_consistency(decision: object) -> None:
    with pytest.raises(ValueError):
        decision()  # type: ignore[operator]


@pytest.mark.parametrize(
    "decision",
    [
        lambda: CoordinatorDecision(
            CoordinatorRoute.START_WRITING_TASK,
            0.8,
            task_input=" ",
        ),
        lambda: CoordinatorDecision(
            CoordinatorRoute.REVISE_TASK,
            0.8,
            revision_instructions=" ",
        ),
    ],
)
def test_route_payloads_reject_blank_text(decision: object) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        decision()  # type: ignore[operator]
