"""Tests for the versioned editorial graph state contract."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from editorial_team.graphs import (
    EDITORIAL_GRAPH_STATE_VERSION,
    EditorialGraphStateV1,
    GraphStateVersionError,
    validate_graph_state_version,
)


def test_state_contract_has_required_version_and_invocation_kind() -> None:
    hints = get_type_hints(EditorialGraphStateV1)

    assert EditorialGraphStateV1.__required_keys__ == {
        "state_version",
        "invocation_kind",
    }
    assert {
        "conversation",
        "turn_conversation",
        "decision",
        "writing_task",
        "editorial_result",
        "assistant_messages",
    } <= hints.keys()


def test_version_one_state_is_accepted() -> None:
    state: EditorialGraphStateV1 = {
        "state_version": EDITORIAL_GRAPH_STATE_VERSION,
        "invocation_kind": "conversation",
    }

    validate_graph_state_version(state)


@pytest.mark.parametrize("version", [None, True, 0, 2, "1"])
def test_missing_or_unsupported_state_version_is_rejected(version: object) -> None:
    state = {"invocation_kind": "conversation"}
    if version is not None:
        state["state_version"] = version

    with pytest.raises(GraphStateVersionError, match="version"):
        validate_graph_state_version(state)  # type: ignore[arg-type]
