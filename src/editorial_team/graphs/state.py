"""Typed, versioned state shared by future editorial graphs."""

from typing import Literal, Required, TypedDict

from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import (
    CriticReport,
    EditorialResult,
    WritingTask,
)
from editorial_team.domain.routing import CoordinatorDecision

EDITORIAL_GRAPH_STATE_VERSION = 1
GraphInvocationKind = Literal["conversation", "external_brief"]


class EditorialGraphStateV1(TypedDict, total=False):
    """Checkpointable state contract for version one of the future graphs.

    Required entry metadata is marked explicitly. The remaining fields are
    populated by later graph nodes and intentionally use the existing immutable,
    provider-neutral domain models.
    """

    state_version: Required[Literal[1]]
    invocation_kind: Required[GraphInvocationKind]
    conversation_id: str
    input_text: str
    prior_conversation: ConversationState
    user_message: Message
    decision: CoordinatorDecision
    talker_response: str
    writing_task: WritingTask
    writer_output: str
    critic_report: CriticReport
    working_draft: str
    editorial_result: EditorialResult
    assistant_contents: tuple[str, ...]
    assistant_messages: tuple[Message, ...]
    routed_conversation: ConversationState
    completed_conversation: ConversationState


class GraphStateVersionError(ValueError):
    """The supplied graph state is absent or uses an unsupported version."""


def validate_graph_state_version(state: EditorialGraphStateV1) -> None:
    """Reject state that is not explicitly versioned as version one."""

    if not isinstance(state, dict):
        raise GraphStateVersionError("Graph state must be an object")
    version = state.get("state_version")
    if type(version) is not int or version != EDITORIAL_GRAPH_STATE_VERSION:
        raise GraphStateVersionError("Graph state version is unsupported")
