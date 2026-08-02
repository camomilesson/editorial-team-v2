"""Fail-closed placeholder nodes for graph topology scaffolding."""

from __future__ import annotations

from collections.abc import Callable

from editorial_team.graphs.state import EditorialGraphStateV1

GraphNode = Callable[[EditorialGraphStateV1], EditorialGraphStateV1]


class GraphScaffoldNotImplementedError(RuntimeError):
    """A disconnected graph scaffold was invoked before implementation."""


def placeholder_node(responsibility: str) -> GraphNode:
    """Create a named node that documents its future role and fails closed."""

    def node(state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        del state
        raise GraphScaffoldNotImplementedError(
            f"Graph node '{responsibility}' is not implemented"
        )

    node.__name__ = responsibility
    node.__doc__ = f"Placeholder for the future {responsibility} responsibility."
    return node
