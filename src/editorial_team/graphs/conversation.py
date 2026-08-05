"""Authoritative LangGraph conversation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from editorial_team.agents.protocols import Critic, Editor, Writer
from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier
from editorial_team.conversation.formatting import (
    format_critic_report,
    format_editor_message,
    format_talker_message,
    format_writer_message,
)
from editorial_team.conversation.protocols import Coordinator, Talker
from editorial_team.domain.conversation import ConversationState, Message, MessageRole
from editorial_team.domain.editorial import (
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.errors import ServiceError
from editorial_team.graphs.editorial import EditorialGraphError, build_editorial_subgraph
from editorial_team.graphs.state import EditorialGraphStateV1, validate_graph_state_version
from editorial_team.tracing import (
    current_trace_stage,
    error_category,
    set_trace_stage,
    trace_event,
)

IdentifierGenerator = Callable[[], str]
UtcClock = Callable[[], datetime]
ParentRoute = Literal["chat", "start_writing_task", "revise_task"]


class ConversationGraphError(ServiceError):
    """A sanitized failure from the conversation graph."""


class _ConversationNodes:
    def __init__(
        self,
        *,
        coordinator: Coordinator,
        talker: Talker,
        editorial_graph: Any,
        identifier_generator: IdentifierGenerator,
        clock: UtcClock,
        max_recent_messages: int,
    ) -> None:
        self._coordinator = coordinator
        self._talker = talker
        self._editorial_graph = editorial_graph
        self._identifier_generator = identifier_generator
        self._clock = clock
        self._max_recent_messages = max_recent_messages

    def validate_and_prepare_turn(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        validate_graph_state_version(state)
        try:
            conversation_id = validate_identifier(state.get("conversation_id"), "conversation_id")  # type: ignore[arg-type]
            text = require_non_blank(state.get("input_text"), "input_text")  # type: ignore[arg-type]
        except ValueError:
            raise ConversationGraphError("Invalid conversation input") from None
        conversation = state.get("conversation")
        if conversation is None:
            conversation = ConversationState(conversation_id)
        if (
            not isinstance(conversation, ConversationState)
            or conversation.conversation_id != conversation_id
        ):
            raise ConversationGraphError("Conversation state is invalid")
        user_message = self._message(conversation_id, MessageRole.USER, text)
        turn_conversation = replace(
            conversation,
            recent_messages=(*conversation.recent_messages, user_message),
        )
        return {
            "turn_conversation": turn_conversation,
            "user_message": user_message,
            "decision": None,
            "talker_response": None,
            "writing_task": None,
            "writer_output": None,
            "critic_report": None,
            "working_draft": None,
            "editorial_result": None,
            "assistant_messages": None,
        }

    def coordinator(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        conversation, user_message = self._turn(state)
        set_trace_stage("coordinator")
        trace_event(
            "coordinator_started",
            active_task=conversation.active_task is not None,
            task_status=None
            if conversation.active_task is None
            else conversation.active_task.status,
        )
        try:
            decision = self._coordinator.decide(conversation, user_message)
        except Exception as exc:
            trace_event(
                "coordinator_failed",
                stage="coordinator",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationGraphError("Coordinator failed") from None
        if not isinstance(decision, CoordinatorDecision):
            raise ConversationGraphError("Coordinator returned an invalid decision")
        try:
            CoordinatorDecision(
                decision.route,
                decision.confidence,
                task_input=decision.task_input,
                revision_instructions=decision.revision_instructions,
            )
        except (TypeError, ValueError):
            raise ConversationGraphError("Coordinator returned an invalid decision") from None
        trace_event("coordinator_completed", route=decision.route, outcome="completed")
        trace_event("route_started", route=decision.route)
        return {"decision": decision}

    def talker(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        conversation, user_message = self._turn(state)
        set_trace_stage("talker")
        trace_event("talker_started", stage="talker")
        try:
            response = self._talker.respond(conversation, user_message)
        except Exception as exc:
            trace_event(
                "talker_failed",
                stage="talker",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationGraphError("Talker failed") from None
        try:
            response = require_non_blank(response, "response")
        except ValueError:
            trace_event(
                "talker_failed", stage="talker", outcome="failed", error_category="blank_response"
            )
            raise ConversationGraphError("Talker returned an invalid response") from None
        trace_event("talker_completed", stage="talker", outcome="completed")
        return {"talker_response": response}

    def prepare_new_task(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        set_trace_stage("writing_workflow")
        trace_event("writing_workflow_started", stage="writing_workflow")
        decision = self._decision(state)
        if decision.task_input is None:
            raise ConversationGraphError("Writing task input is invalid")
        conversation, _ = self._turn(state)
        now = self._timestamp()
        return {
            "writing_task": WritingTask(
                self._identifier("task"),
                conversation.conversation_id,
                WritingBrief(decision.task_input),
                WritingTaskStatus.CREATED,
                now,
                now,
            )
        }

    def prepare_revision(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        set_trace_stage("revision_workflow")
        trace_event("revision_workflow_started", stage="revision_workflow")
        conversation, _ = self._turn(state)
        task = conversation.active_task
        if (
            task is None
            or task.status not in {WritingTaskStatus.REVIEWED, WritingTaskStatus.REVISED}
            or task.critic_report is None
            or task.working_draft is None
        ):
            raise ConversationGraphError("No writing task is available for revision")
        decision = self._decision(state)
        if decision.revision_instructions is None:
            raise ConversationGraphError("Revision instructions are invalid")
        try:
            brief = replace(
                task.brief,
                instructions=(*task.brief.instructions, decision.revision_instructions),
            )
            return {"writing_task": replace(task, brief=brief)}
        except (TypeError, ValueError):
            raise ConversationGraphError("Revision task is invalid") from None

    def editorial_subgraph(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        try:
            output = self._editorial_graph.invoke(state)
        except EditorialGraphError as exc:
            trace_event(
                "writing_workflow_failed",
                stage=current_trace_stage(),
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationGraphError("Writing workflow failed") from None
        result = output.get("editorial_result")
        if not isinstance(result, EditorialResult):
            raise ConversationGraphError("Writing workflow returned an invalid result")
        return {
            "writer_output": result.writer_output,
            "critic_report": result.critic_report,
            "working_draft": result.working_draft,
            "editorial_result": result,
        }

    def finalize_task(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        conversation, _ = self._turn(state)
        decision = self._decision(state)
        task = state.get("writing_task")
        result = state.get("editorial_result")
        if not isinstance(task, WritingTask) or not isinstance(result, EditorialResult):
            raise ConversationGraphError("Writing workflow returned an invalid result")
        status = (
            WritingTaskStatus.REVISED if result.revision_applied else WritingTaskStatus.REVIEWED
        )
        if decision.route is CoordinatorRoute.START_WRITING_TASK:
            active_task = WritingTask(
                task.id,
                task.conversation_id,
                task.brief,
                status,
                task.created_at,
                self._timestamp(),
                result.working_draft,
                result.critic_report,
            )
            stage = "writing_workflow"
        elif decision.route is CoordinatorRoute.REVISE_TASK:
            active_task = replace(
                task,
                status=status,
                updated_at=self._timestamp(),
                working_draft=result.working_draft,
                critic_report=result.critic_report,
            )
            stage = "revision_workflow"
        else:
            raise ConversationGraphError("Coordinator returned an unsupported route")
        set_trace_stage(stage)
        trace_event(
            f"{stage}_completed",
            stage=stage,
            outcome="completed",
            critic_verdict=result.critic_report.verdict,
            revision_applied=result.revision_applied,
        )
        return {"turn_conversation": replace(conversation, active_task=active_task)}

    def finalize_turn(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        conversation, _ = self._turn(state)
        decision = self._decision(state)
        if decision.route is CoordinatorRoute.CHAT:
            response = state.get("talker_response")
            if not isinstance(response, str):
                raise ConversationGraphError("Talker returned an invalid response")
            contents = (format_talker_message(response),)
        else:
            result = state.get("editorial_result")
            if not isinstance(result, EditorialResult):
                raise ConversationGraphError("Writing workflow returned an invalid result")
            contents = (
                format_writer_message(result.writer_output),
                format_critic_report(result.critic_report),
                format_editor_message(result),
            )
        messages = tuple(
            self._message(conversation.conversation_id, MessageRole.ASSISTANT, content)
            for content in contents
        )
        completed = replace(
            conversation,
            recent_messages=(*conversation.recent_messages, *messages)[
                -self._max_recent_messages :
            ],
        )
        return {
            "conversation": completed,
            "assistant_messages": messages,
            "input_text": None,
            "turn_conversation": None,
            "user_message": None,
            "decision": None,
            "talker_response": None,
            "writing_task": None,
            "writer_output": None,
            "critic_report": None,
            "working_draft": None,
            "editorial_result": None,
        }

    @staticmethod
    def _turn(state: EditorialGraphStateV1) -> tuple[ConversationState, Message]:
        conversation = state.get("turn_conversation")
        message = state.get("user_message")
        if not isinstance(conversation, ConversationState) or not isinstance(message, Message):
            raise ConversationGraphError("Invalid conversation input")
        return conversation, message

    @staticmethod
    def _decision(state: EditorialGraphStateV1) -> CoordinatorDecision:
        decision = state.get("decision")
        if not isinstance(decision, CoordinatorDecision):
            raise ConversationGraphError("Coordinator returned an invalid decision")
        return decision

    def _identifier(self, kind: str) -> str:
        try:
            return validate_identifier(self._identifier_generator(), f"{kind}_id")
        except Exception:
            raise ConversationGraphError("Identifier generation failed") from None

    def _timestamp(self) -> datetime:
        try:
            return require_utc_timestamp(self._clock(), "timestamp")
        except Exception:
            raise ConversationGraphError("Clock failed") from None

    def _message(self, conversation_id: str, role: MessageRole, content: str) -> Message:
        try:
            return Message(
                self._identifier("message"), conversation_id, role, content, self._timestamp()
            )
        except ConversationGraphError:
            raise
        except (TypeError, ValueError):
            raise ConversationGraphError("Message creation failed") from None


def _route_coordinator_decision(state: EditorialGraphStateV1) -> ParentRoute:
    decision = state.get("decision")
    if not isinstance(decision, CoordinatorDecision):
        raise ConversationGraphError("Coordinator returned an invalid decision")
    return decision.route.value


def build_parent_graph(
    *,
    coordinator: Coordinator,
    talker: Talker,
    writer: Writer,
    critic: Critic,
    editor: Editor,
    identifier_generator: IdentifierGenerator,
    clock: UtcClock,
    max_recent_messages: int,
) -> StateGraph[EditorialGraphStateV1]:
    """Build the complete authoritative conversation graph."""

    editorial_graph = build_editorial_subgraph(
        writer=writer, critic=critic, editor=editor
    ).compile()
    nodes = _ConversationNodes(
        coordinator=coordinator,
        talker=talker,
        editorial_graph=editorial_graph,
        identifier_generator=identifier_generator,
        clock=clock,
        max_recent_messages=max_recent_messages,
    )
    graph = StateGraph(EditorialGraphStateV1)
    graph.add_node("validate_and_prepare_turn", nodes.validate_and_prepare_turn)
    graph.add_node("coordinator", nodes.coordinator)
    graph.add_node("talker", nodes.talker)
    graph.add_node("prepare_new_task", nodes.prepare_new_task)
    graph.add_node("prepare_revision", nodes.prepare_revision)
    graph.add_node("editorial_subgraph", nodes.editorial_subgraph)
    graph.add_node("finalize_task", nodes.finalize_task)
    graph.add_node("finalize_turn", nodes.finalize_turn)
    graph.add_edge(START, "validate_and_prepare_turn")
    graph.add_edge("validate_and_prepare_turn", "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        _route_coordinator_decision,
        {
            "chat": "talker",
            "start_writing_task": "prepare_new_task",
            "revise_task": "prepare_revision",
        },
    )
    graph.add_edge("talker", "finalize_turn")
    graph.add_edge("prepare_new_task", "editorial_subgraph")
    graph.add_edge("prepare_revision", "editorial_subgraph")
    graph.add_edge("editorial_subgraph", "finalize_task")
    graph.add_edge("finalize_task", "finalize_turn")
    graph.add_edge("finalize_turn", END)
    return graph
