"""Authoritative LangGraph conversation orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from editorial_team.agents.coordinator import ToolCallingCoordinator
from editorial_team.agents.parsing import parse_coordinator_decision
from editorial_team.agents.prompts import retrieval_coordinator_prompt
from editorial_team.agents.protocols import Critic, Editor, Writer
from editorial_team.artifacts.models import (
    ArtifactProducer,
    EditorialArtifact,
    artifact_id_for,
    content_sha256,
)
from editorial_team.artifacts.protocols import ArtifactStore
from editorial_team.artifacts.retrieval import HybridRetriever
from editorial_team.artifacts.retrieval_types import RetrievedDraft
from editorial_team.artifacts.tools import build_editorial_retrieval_tools
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
ParentRoute = Literal["tools", "chat", "start_writing_task", "revise_task"]
MAX_COORDINATOR_TOOL_ROUNDS = 6


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
        artifact_store: ArtifactStore,
        tool_coordinator: ToolCallingCoordinator | None,
        retriever: HybridRetriever | None,
        user_timezone: str,
    ) -> None:
        self._coordinator = coordinator
        self._talker = talker
        self._editorial_graph = editorial_graph
        self._identifier_generator = identifier_generator
        self._clock = clock
        self._max_recent_messages = max_recent_messages
        self._artifact_store = artifact_store
        self._tool_coordinator = tool_coordinator
        self._retriever = retriever
        self._user_timezone = user_timezone

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
            "coordinator_messages": None,
            "coordinator_tool_steps": 0,
            "coordinator_search_completed": False,
            "retrieved_draft": None,
        }

    def coordinator_agent(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
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
            if self._tool_coordinator is None:
                decision = self._coordinator.decide(conversation, user_message)
                response = None
            else:
                messages = state.get("coordinator_messages")
                if messages is None:
                    now = self._timestamp()
                    zone = ZoneInfo(self._user_timezone)
                    prompt = retrieval_coordinator_prompt(
                        conversation,
                        user_message,
                        current_local_datetime=now.astimezone(zone).isoformat(),
                        user_timezone=self._user_timezone,
                        current_utc_datetime=now.isoformat(),
                    )
                    messages = (
                        SystemMessage(content=prompt),
                        HumanMessage(content=user_message.content),
                    )
                tools = self._scoped_tools(conversation.conversation_id)
                response = self._tool_coordinator.respond(messages, tools)
                messages = (*messages, response)
                if response.tool_calls:
                    return {"coordinator_messages": messages}
                decision = parse_coordinator_decision(self._message_text(response))
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
                talker_context=decision.talker_context,
            )
        except (TypeError, ValueError):
            raise ConversationGraphError("Coordinator returned an invalid decision") from None
        trace_event("coordinator_completed", route=decision.route, outcome="completed")
        retrieved = state.get("retrieved_draft")
        if (
            isinstance(retrieved, RetrievedDraft)
            and decision.route is not CoordinatorRoute.START_WRITING_TASK
        ):
            raise ConversationGraphError("Retrieved draft requires a new writing task")
        trace_event("route_started", route=decision.route)
        return {
            "decision": decision,
            "coordinator_messages": (
                state.get("coordinator_messages") if response is None else messages
            ),
        }

    def coordinator_tools(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Execute one LLM-selected round through scoped LangChain tool runnables."""

        conversation, _ = self._turn(state)
        messages = state.get("coordinator_messages")
        if (
            not isinstance(messages, tuple)
            or not messages
            or not isinstance(messages[-1], AIMessage)
        ):
            raise ConversationGraphError("Coordinator tool state is invalid")
        steps = state.get("coordinator_tool_steps", 0)
        if not isinstance(steps, int) or steps >= MAX_COORDINATOR_TOOL_ROUNDS:
            raise ConversationGraphError("Coordinator tool limit exceeded")
        tools = {tool.name: tool for tool in self._scoped_tools(conversation.conversation_id)}
        tool_messages: list[ToolMessage] = []
        retrieved = state.get("retrieved_draft")
        search_completed = state.get("coordinator_search_completed") is True
        for call in messages[-1].tool_calls:
            name = call.get("name")
            tool = tools.get(name)
            if tool is None:
                output = {
                    "ok": False,
                    "error": {
                        "type": "unknown_tool",
                        "message": "The tool is unavailable",
                    },
                }
            elif name == "get_draft" and not search_completed:
                output = {
                    "ok": False,
                    "error": {
                        "type": "search_required",
                        "message": "Search the corpus before selecting a draft",
                    },
                }
            else:
                try:
                    output = tool.invoke(call.get("args", {}))
                except Exception:
                    output = {
                        "ok": False,
                        "error": {
                            "type": "invalid_tool_arguments",
                            "message": "The tool arguments are invalid",
                        },
                    }
            if (
                name == "search_corpus"
                and isinstance(output, dict)
                and output.get("ok") is True
            ):
                search_completed = True
            if name == "get_draft" and isinstance(output, dict) and output.get("ok") is True:
                retrieved = self._retrieved_draft(output, conversation.conversation_id)
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(output, ensure_ascii=False, allow_nan=False),
                    tool_call_id=str(call.get("id", "missing-call-id")),
                    name=str(name),
                )
            )
        return {
            "coordinator_messages": (*messages, *tool_messages),
            "coordinator_tool_steps": steps + 1,
            "coordinator_search_completed": search_completed,
            "retrieved_draft": retrieved,
        }

    def talker(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        conversation, user_message = self._turn(state)
        set_trace_stage("talker")
        trace_event("talker_started", stage="talker")
        try:
            context = self._decision(state).talker_context
            if context is None:
                response = self._talker.respond(conversation, user_message)
            else:
                response = self._talker.respond(conversation, user_message, context)
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
        task_id = self._identifier("task")
        retrieved = state.get("retrieved_draft")
        if state.get("coordinator_tool_steps", 0) and not isinstance(retrieved, RetrievedDraft):
            raise ConversationGraphError("A complete historical draft was not retrieved")
        return {
            "writing_task": WritingTask(
                task_id,
                conversation.conversation_id,
                WritingBrief(decision.task_input),
                WritingTaskStatus.CREATED,
                now,
                now,
                None if retrieved is None else retrieved.content,
            ),
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
            now = self._timestamp()
            brief = replace(
                task.brief,
                instructions=(*task.brief.instructions, decision.revision_instructions),
            )
            return {
                "writing_task": replace(
                    task,
                    id=self._identifier("task"),
                    brief=brief,
                    created_at=now,
                    updated_at=now,
                ),
            }
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

    def persist_editorial_artifacts(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Persist the complete successful Writer/optional Editor output set."""

        completed_state = self._completed_turn_state(state)
        conversation, _ = self._turn(state)
        result = state.get("editorial_result")
        task = state.get("writing_task")
        decision = self._decision(state)
        user_request = (
            decision.task_input
            if decision.route is CoordinatorRoute.START_WRITING_TASK
            else decision.revision_instructions
        )
        if (
            not isinstance(result, EditorialResult)
            or not isinstance(task, WritingTask)
            or not isinstance(user_request, str)
        ):
            raise ConversationGraphError("Editorial artifact input is invalid")
        artifacts = [
            self._artifact(
                task_id=task.id,
                producer=ArtifactProducer.WRITER,
                created_at=task.created_at,
                conversation_id=conversation.conversation_id,
                user_request=user_request,
                content=result.writer_output,
            )
        ]
        if result.revision_applied:
            artifacts.append(
                self._artifact(
                    task_id=task.id,
                    producer=ArtifactProducer.EDITOR,
                    created_at=task.created_at,
                    conversation_id=conversation.conversation_id,
                    user_request=user_request,
                    content=result.working_draft,
                )
            )
        try:
            self._artifact_store.save_run(tuple(artifacts))
        except Exception as exc:
            trace_event(
                "artifact_persistence_failed",
                stage="artifact_persistence",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationGraphError("Editorial artifacts could not be saved") from None
        trace_event(
            "artifact_persistence_completed",
            stage="artifact_persistence",
            outcome="completed",
            artifact_count=len(artifacts),
        )
        return completed_state

    def finalize_turn(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Finalize a non-artifact chat turn."""

        return self._completed_turn_state(state)

    def _completed_turn_state(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
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
            "coordinator_messages": None,
            "coordinator_tool_steps": None,
            "coordinator_search_completed": None,
            "retrieved_draft": None,
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

    @staticmethod
    def _artifact(
        *,
        task_id: str,
        producer: ArtifactProducer,
        created_at: datetime,
        conversation_id: str,
        user_request: str,
        content: str,
    ) -> EditorialArtifact:
        try:
            return EditorialArtifact(
                artifact_id=artifact_id_for(task_id, producer),
                task_id=task_id,
                producer=producer,
                created_at=created_at,
                conversation_id=conversation_id,
                user_request=user_request,
                content=content,
                content_sha256=content_sha256(content),
            )
        except (TypeError, ValueError):
            raise ConversationGraphError("Editorial artifact input is invalid") from None

    def _scoped_tools(self, conversation_id: str) -> tuple[Any, Any]:
        if self._retriever is None:
            raise ConversationGraphError("Coordinator retrieval is unavailable")
        return build_editorial_retrieval_tools(
            retriever=self._retriever,
            conversation_id=conversation_id,
        )

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        if not isinstance(message.content, str) or not message.content.strip():
            raise ConversationGraphError("Coordinator returned an invalid decision")
        return message.content

    @staticmethod
    def _retrieved_draft(output: dict[str, Any], conversation_id: str) -> RetrievedDraft:
        from editorial_team.contracts.common import parse_utc_timestamp

        try:
            data = output["data"]
            return RetrievedDraft(
                artifact_id=data["artifact_id"],
                task_id=data["task_id"],
                producer=ArtifactProducer(data["producer"]),
                created_at=parse_utc_timestamp(data["created_at"], "created_at"),
                conversation_id=conversation_id,
                user_request=data["user_request"],
                content=data["content"],
            )
        except (KeyError, TypeError, ValueError):
            raise ConversationGraphError("Retrieved draft is invalid") from None


def _route_coordinator_decision(state: EditorialGraphStateV1) -> ParentRoute:
    messages = state.get("coordinator_messages")
    if (
        isinstance(messages, tuple)
        and messages
        and isinstance(messages[-1], AIMessage)
        and messages[-1].tool_calls
    ):
        return "tools"
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
    artifact_store: ArtifactStore,
    tool_coordinator: ToolCallingCoordinator | None = None,
    retriever: HybridRetriever | None = None,
    user_timezone: str = "Europe/Madrid",
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
        artifact_store=artifact_store,
        tool_coordinator=tool_coordinator,
        retriever=retriever,
        user_timezone=user_timezone,
    )
    graph = StateGraph(EditorialGraphStateV1)
    graph.add_node("validate_and_prepare_turn", nodes.validate_and_prepare_turn)
    graph.add_node("coordinator_agent", nodes.coordinator_agent)
    graph.add_node("coordinator_tools", nodes.coordinator_tools)
    graph.add_node("talker", nodes.talker)
    graph.add_node("prepare_new_task", nodes.prepare_new_task)
    graph.add_node("prepare_revision", nodes.prepare_revision)
    graph.add_node("editorial_subgraph", nodes.editorial_subgraph)
    graph.add_node("finalize_task", nodes.finalize_task)
    graph.add_node("persist_editorial_artifacts", nodes.persist_editorial_artifacts)
    graph.add_node("finalize_turn", nodes.finalize_turn)
    graph.add_edge(START, "validate_and_prepare_turn")
    graph.add_edge("validate_and_prepare_turn", "coordinator_agent")
    graph.add_conditional_edges(
        "coordinator_agent",
        _route_coordinator_decision,
        {
            "tools": "coordinator_tools",
            "chat": "talker",
            "start_writing_task": "prepare_new_task",
            "revise_task": "prepare_revision",
        },
    )
    graph.add_edge("coordinator_tools", "coordinator_agent")
    graph.add_edge("talker", "finalize_turn")
    graph.add_edge("prepare_new_task", "editorial_subgraph")
    graph.add_edge("prepare_revision", "editorial_subgraph")
    graph.add_edge("editorial_subgraph", "finalize_task")
    graph.add_edge("finalize_task", "persist_editorial_artifacts")
    graph.add_edge("persist_editorial_artifacts", END)
    graph.add_edge("finalize_turn", END)
    return graph
