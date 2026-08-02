"""Deterministic conversational application service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any

from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier
from editorial_team.conversation.formatting import (
    format_critic_report,
    format_editor_message,
    format_talker_message,
    format_writer_message,
)
from editorial_team.conversation.protocols import (
    ConversationStateStore,
    Coordinator,
    Talker,
    WritingWorkflowRunner,
)
from editorial_team.domain.conversation import ConversationState, Message, MessageRole
from editorial_team.domain.editorial import (
    CriticReport,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision
from editorial_team.errors import ServiceError
from editorial_team.tracing import (
    current_trace_stage,
    error_category,
    set_trace_stage,
    trace_event,
)

IdentifierGenerator = Callable[[], str]
UtcClock = Callable[[], datetime]


class ConversationServiceError(ServiceError):
    """A sanitized failure at the conversation application boundary."""


class ConversationService:
    """Process one user message atomically through a selected route."""

    def __init__(
        self,
        *,
        coordinator: Coordinator,
        talker: Talker,
        workflow: WritingWorkflowRunner,
        store: ConversationStateStore,
        identifier_generator: IdentifierGenerator,
        clock: UtcClock,
        max_recent_messages: int,
        graph_runner: Any | None = None,
        graph_checkpointer: Any | None = None,
    ) -> None:
        if (
            isinstance(max_recent_messages, bool)
            or not isinstance(max_recent_messages, int)
            or max_recent_messages <= 0
        ):
            raise ValueError("max_recent_messages must be a positive integer")
        self._coordinator = coordinator
        self._talker = talker
        self._workflow = workflow
        self._store = store
        self._identifier_generator = identifier_generator
        self._clock = clock
        self._max_recent_messages = max_recent_messages
        if graph_runner is None:
            from editorial_team.graphs import (
                build_parent_graph,
                create_in_memory_checkpointer,
            )

            graph_checkpointer = create_in_memory_checkpointer()
            graph_runner = build_parent_graph(
                coordinator=coordinator,
                talker=talker,
                workflow=workflow,
                store=store,
                identifier_generator=identifier_generator,
                clock=clock,
                max_recent_messages=max_recent_messages,
            ).compile(checkpointer=graph_checkpointer)
        self._graph_runner = graph_runner
        self._graph_checkpointer = graph_checkpointer

    def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
        """Process one turn and return only its assistant messages."""

        try:
            graph_state = self._graph_runner.invoke(
                {
                    "state_version": 1,
                    "invocation_kind": "conversation",
                    "conversation_id": conversation_id,
                    "input_text": text,
                },
                {
                    "configurable": {
                        "thread_id": f"editorial:v1:{conversation_id}",
                    }
                },
            )
        except ConversationServiceError as exc:
            raise ConversationServiceError(str(exc)) from None
        assistant_messages = graph_state.get("assistant_messages")
        completed_state = graph_state.get("completed_conversation")
        if (
            not isinstance(assistant_messages, tuple)
            or not assistant_messages
            or not all(isinstance(message, Message) for message in assistant_messages)
            or not isinstance(completed_state, ConversationState)
        ):
            raise ConversationServiceError("Conversation graph returned an invalid result")
        self._save_state(completed_state)
        return assistant_messages

    def process_brief(self, brief: str) -> EditorialResult:
        """Run a validated standalone brief through the shared writing workflow."""

        try:
            brief = require_non_blank(brief, "brief")
        except ValueError:
            raise ConversationServiceError("Invalid writing brief") from None
        timestamp = self._timestamp()
        task = WritingTask(
            id=self._identifier("task"),
            conversation_id=self._identifier("external"),
            brief=WritingBrief(brief),
            status=WritingTaskStatus.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        set_trace_stage("writing_workflow")
        trace_event("writing_workflow_started", stage="writing_workflow")
        result = self._execute_workflow(task)
        trace_event(
            "writing_workflow_completed",
            stage="writing_workflow",
            outcome="completed",
            revision_applied=result.revision_applied,
        )
        return result

    def _load_state(self, conversation_id: str) -> ConversationState:
        try:
            state = self._store.load(conversation_id)
        except Exception:
            raise ConversationServiceError("Conversation state could not be loaded") from None
        if state is None:
            return ConversationState(conversation_id)
        if not isinstance(state, ConversationState) or state.conversation_id != conversation_id:
            raise ConversationServiceError("Conversation state is invalid")
        return state

    def _save_state(self, state: ConversationState) -> None:
        try:
            self._store.save(state)
        except Exception:
            raise ConversationServiceError("Conversation state could not be saved") from None

    def _decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        set_trace_stage("coordinator")
        trace_event(
            "coordinator_started",
            active_task=state.active_task is not None,
            task_status=(
                None if state.active_task is None else state.active_task.status
            ),
        )
        try:
            decision = self._coordinator.decide(state, user_message)
        except Exception as exc:
            trace_event(
                "coordinator_failed",
                stage="coordinator",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationServiceError("Coordinator failed") from None
        if not isinstance(decision, CoordinatorDecision):
            trace_event(
                "coordinator_failed",
                stage="coordinator",
                outcome="failed",
                error_category="schema_validation_failure",
            )
            raise ConversationServiceError("Coordinator returned an invalid decision")
        try:
            CoordinatorDecision(
                route=decision.route,
                confidence=decision.confidence,
                task_input=decision.task_input,
                revision_instructions=decision.revision_instructions,
            )
        except (TypeError, ValueError):
            trace_event(
                "coordinator_failed",
                stage="coordinator",
                outcome="failed",
                error_category="domain_consistency_failure",
            )
            raise ConversationServiceError("Coordinator returned an invalid decision") from None
        trace_event(
            "coordinator_completed",
            route=decision.route,
            outcome="completed",
        )
        return decision

    def _chat(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> tuple[ConversationState, tuple[str, ...]]:
        set_trace_stage("talker")
        response = self._talk(state, user_message)
        return state, (format_talker_message(response),)

    def _start_task(
        self,
        state: ConversationState,
        decision: CoordinatorDecision,
    ) -> tuple[ConversationState, tuple[str, ...]]:
        set_trace_stage("writing_workflow")
        trace_event("writing_workflow_started", stage="writing_workflow")
        if decision.task_input is None:
            raise ConversationServiceError("Writing task input is invalid")
        timestamp = self._timestamp()
        task = WritingTask(
            id=self._identifier("task"),
            conversation_id=state.conversation_id,
            brief=WritingBrief(decision.task_input),
            status=WritingTaskStatus.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        result = self._execute_workflow(task)
        set_trace_stage("writing_workflow")
        trace_event(
            "writing_workflow_completed",
            stage="writing_workflow",
            outcome="completed",
            critic_verdict=result.critic_report.verdict,
            revision_applied=result.revision_applied,
        )
        active_task = WritingTask(
            id=task.id,
            conversation_id=task.conversation_id,
            brief=task.brief,
            status=self._completed_status(result),
            created_at=task.created_at,
            updated_at=self._timestamp(),
            working_draft=result.working_draft,
            critic_report=result.critic_report,
        )
        routed_state = replace(state, active_task=active_task)
        return routed_state, self._writing_messages(result)

    def _revise_task(
        self,
        state: ConversationState,
        decision: CoordinatorDecision,
    ) -> tuple[ConversationState, tuple[str, ...]]:
        set_trace_stage("revision_workflow")
        trace_event("revision_workflow_started", stage="revision_workflow")
        task = self._require_latest_task(state)
        if decision.revision_instructions is None:
            raise ConversationServiceError("Revision instructions are invalid")
        try:
            brief = replace(
                task.brief,
                instructions=(*task.brief.instructions, decision.revision_instructions),
            )
            workflow_task = replace(
                task,
                brief=brief,
            )
        except (TypeError, ValueError):
            raise ConversationServiceError("Revision task is invalid") from None

        result = self._execute_workflow(workflow_task)
        set_trace_stage("revision_workflow")
        trace_event(
            "revision_workflow_completed",
            stage="revision_workflow",
            outcome="completed",
            critic_verdict=result.critic_report.verdict,
            revision_applied=result.revision_applied,
        )
        active_task = replace(
            workflow_task,
            status=self._completed_status(result),
            working_draft=result.working_draft,
            critic_report=result.critic_report,
            updated_at=self._timestamp(),
        )
        routed_state = replace(state, active_task=active_task)
        return routed_state, self._writing_messages(result)

    def _require_latest_task(self, state: ConversationState) -> WritingTask:
        task = state.active_task
        if (
            task is None
            or task.status not in {WritingTaskStatus.REVIEWED, WritingTaskStatus.REVISED}
            or task.critic_report is None
            or task.working_draft is None
        ):
            raise ConversationServiceError("No writing task is available for revision")
        return task

    def _execute_workflow(self, task: WritingTask) -> EditorialResult:
        try:
            result = self._workflow.execute(task)
        except Exception as exc:
            trace_event(
                "writing_workflow_failed",
                stage=current_trace_stage(),
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationServiceError("Writing workflow failed") from None
        set_trace_stage("writing_workflow")
        if not isinstance(result, EditorialResult):
            trace_event(
                "writing_workflow_failed",
                stage="writing_workflow",
                outcome="failed",
                error_category="schema_validation_failure",
            )
            raise ConversationServiceError("Writing workflow returned an invalid result")
        try:
            CriticReport(
                verdict=result.critic_report.verdict,
                summary=result.critic_report.summary,
                issues=result.critic_report.issues,
            )
            EditorialResult(
                writer_output=result.writer_output,
                critic_report=result.critic_report,
                working_draft=result.working_draft,
                revision_applied=result.revision_applied,
            )
        except (TypeError, ValueError):
            trace_event(
                "writing_workflow_failed",
                stage="writing_workflow",
                outcome="failed",
                error_category="domain_consistency_failure",
            )
            raise ConversationServiceError("Writing workflow returned an invalid result") from None
        return result

    def _talk(self, state: ConversationState, user_message: Message) -> str:
        set_trace_stage("talker")
        trace_event("talker_started", stage="talker")
        try:
            response = self._talker.respond(state, user_message)
        except Exception as exc:
            trace_event(
                "talker_failed",
                stage="talker",
                outcome="failed",
                error_category=error_category(exc),
            )
            raise ConversationServiceError("Talker failed") from None
        try:
            require_non_blank(response, "response")
        except ValueError:
            trace_event(
                "talker_failed",
                stage="talker",
                outcome="failed",
                error_category="blank_response",
            )
            raise ConversationServiceError("Talker returned an invalid response") from None
        trace_event("talker_completed", stage="talker", outcome="completed")
        return response

    def _writing_messages(self, result: EditorialResult) -> tuple[str, ...]:
        return (
            format_writer_message(result.writer_output),
            format_critic_report(result.critic_report),
            format_editor_message(result),
        )

    @staticmethod
    def _completed_status(result: EditorialResult) -> WritingTaskStatus:
        return (
            WritingTaskStatus.REVISED
            if result.revision_applied
            else WritingTaskStatus.REVIEWED
        )

    def _identifier(self, kind: str) -> str:
        try:
            value = self._identifier_generator()
            return validate_identifier(value, f"{kind}_id")
        except Exception:
            raise ConversationServiceError("Identifier generation failed") from None

    def _timestamp(self) -> datetime:
        try:
            value = self._clock()
            return require_utc_timestamp(value, "timestamp")
        except Exception:
            raise ConversationServiceError("Clock failed") from None

    def _message(
        self,
        *,
        conversation_id: str,
        role: MessageRole,
        content: str,
    ) -> Message:
        try:
            return Message(
                id=self._identifier("message"),
                conversation_id=conversation_id,
                role=role,
                content=content,
                created_at=self._timestamp(),
            )
        except ConversationServiceError:
            raise
        except (TypeError, ValueError):
            raise ConversationServiceError("Message creation failed") from None
