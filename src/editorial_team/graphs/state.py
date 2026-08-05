"""Versioned state for the authoritative conversation graph."""

from typing import Literal, Required, TypedDict

from langchain_core.messages import BaseMessage

from editorial_team.artifacts.retrieval_types import RetrievedDraft
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import CriticReport, EditorialResult, WritingTask
from editorial_team.domain.routing import CoordinatorDecision

EDITORIAL_GRAPH_STATE_VERSION = 1
GraphInvocationKind = Literal["conversation"]


class EditorialGraphStateV1(TypedDict, total=False):
    """Durable conversation state plus transient fields for one graph turn.

    ``conversation`` is the sole durable product state. Other optional values exist
    only while a turn executes, except ``assistant_messages`` which is retained as
    the facade's result until the next turn starts.
    """

    state_version: Required[Literal[1]]
    invocation_kind: Required[GraphInvocationKind]
    conversation_id: str
    input_text: str | None
    conversation: ConversationState
    turn_conversation: ConversationState | None
    user_message: Message | None
    decision: CoordinatorDecision | None
    talker_response: str | None
    writing_task: WritingTask | None
    writer_output: str | None
    critic_report: CriticReport | None
    working_draft: str | None
    editorial_result: EditorialResult | None
    assistant_messages: tuple[Message, ...] | None
    coordinator_messages: tuple[BaseMessage, ...] | None
    coordinator_tool_steps: int | None
    coordinator_search_completed: bool | None
    retrieved_draft: RetrievedDraft | None


class GraphStateVersionError(ValueError):
    """The supplied graph state is absent or uses an unsupported version."""


def validate_graph_state_version(state: EditorialGraphStateV1) -> None:
    """Reject state that is not explicitly versioned as version one."""

    if not isinstance(state, dict):
        raise GraphStateVersionError("Graph state must be an object")
    version = state.get("state_version")
    if type(version) is not int or version != EDITORIAL_GRAPH_STATE_VERSION:
        raise GraphStateVersionError("Graph state version is unsupported")
