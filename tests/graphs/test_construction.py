"""Tests for disconnected graph topology scaffolding."""

from __future__ import annotations

from datetime import UTC, datetime

from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.graphs import build_editorial_subgraph, build_parent_graph


class UnusedWriter:
    def write(self, task: object) -> str:
        del task
        raise AssertionError("construction must not invoke Writer")


class UnusedCritic:
    def review(self, task: object, draft: str) -> object:
        del task, draft
        raise AssertionError("construction must not invoke Critic")


class UnusedEditor:
    def revise(self, task: object, draft: str, report: object) -> str:
        del task, draft, report
        raise AssertionError("construction must not invoke Editor")


class StaticCoordinator:
    def __init__(self, decision: CoordinatorDecision) -> None:
        self.decision = decision

    def decide(self, state: object, user_message: object) -> CoordinatorDecision:
        del state, user_message
        return self.decision


class UnusedTalker:
    def respond(self, state: object, user_message: object) -> str:
        del state, user_message
        raise AssertionError("construction must not invoke Talker")


class UnusedArtifactStore:
    def save_run(self, artifacts: object) -> None:
        del artifacts
        raise AssertionError("construction must not persist artifacts")


def parent_builder(
    decision: CoordinatorDecision | None = None,
) -> object:
    return build_parent_graph(
        coordinator=StaticCoordinator(decision or CoordinatorDecision(CoordinatorRoute.CHAT, 1.0)),
        talker=UnusedTalker(),
        identifier_generator=lambda: "message-1",
        clock=lambda: datetime.now(UTC),
        writer=UnusedWriter(),
        critic=UnusedCritic(),
        editor=UnusedEditor(),
        max_recent_messages=20,
        artifact_store=UnusedArtifactStore(),
    )


def _node_names(graph: object) -> set[str]:
    compiled = graph.compile()  # type: ignore[attr-defined]
    return set(compiled.get_graph().nodes)


def test_parent_graph_contains_planned_foundation_nodes() -> None:
    assert _node_names(parent_builder()) == {
        "__start__",
        "validate_and_prepare_turn",
        "coordinator",
        "talker",
        "prepare_new_task",
        "prepare_revision",
        "editorial_subgraph",
        "finalize_task",
        "persist_editorial_artifacts",
        "finalize_turn",
        "__end__",
    }


def test_parent_graph_preserves_planned_topology() -> None:
    graph = parent_builder().compile().get_graph()  # type: ignore[attr-defined]

    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "validate_and_prepare_turn", False),
        ("validate_and_prepare_turn", "coordinator", False),
        ("coordinator", "talker", True),
        ("coordinator", "prepare_new_task", True),
        ("coordinator", "prepare_revision", True),
        ("talker", "finalize_turn", False),
        ("prepare_new_task", "editorial_subgraph", False),
        ("prepare_revision", "editorial_subgraph", False),
        ("editorial_subgraph", "finalize_task", False),
        ("finalize_task", "persist_editorial_artifacts", False),
        ("persist_editorial_artifacts", "__end__", False),
        ("finalize_turn", "__end__", False),
    }


def test_editorial_subgraph_contains_planned_foundation_nodes() -> None:
    builder = build_editorial_subgraph(
        writer=UnusedWriter(),
        critic=UnusedCritic(),
        editor=UnusedEditor(),
    )

    assert _node_names(builder) == {
        "__start__",
        "writer",
        "critic",
        "editor",
        "build_pass_result",
        "build_revised_result",
        "__end__",
    }


def test_editorial_subgraph_preserves_planned_topology() -> None:
    graph = (
        build_editorial_subgraph(
            writer=UnusedWriter(),
            critic=UnusedCritic(),
            editor=UnusedEditor(),
        )
        .compile()
        .get_graph()
    )

    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "writer", False),
        ("writer", "critic", False),
        ("critic", "build_pass_result", True),
        ("critic", "editor", True),
        ("editor", "build_revised_result", False),
        ("build_pass_result", "__end__", False),
        ("build_revised_result", "__end__", False),
    }
