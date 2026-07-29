"""Deterministic conversational application service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from editorial_team.contracts.common import require_non_blank, require_utc_timestamp
from editorial_team.contracts.identity import validate_identifier
from editorial_team.conversation.formatting import (
    format_critic_report,
    format_working_draft,
    request_user_evaluation,
)
from editorial_team.conversation.protocols import (
    ConversationStateStore,
    Coordinator,
    Talker,
    WritingWorkflowRunner,
)
from editorial_team.domain.conversation import (
    ConversationState,
    ConversationStatus,
    Message,
    MessageRole,
)
from editorial_team.domain.editorial import (
    CriticReport,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
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

    def process_message(self, conversation_id: str, text: str) -> tuple[Message, ...]:
        """Process one turn and return only its assistant messages."""

        try:
            conversation_id = validate_identifier(conversation_id, "conversation_id")
            require_non_blank(text, "text")
        except ValueError:
            raise ConversationServiceError("Invalid conversation input") from None

        state = self._load_state(conversation_id)
        user_message = self._message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=text,
        )
        state_with_user = replace(
            state,
            recent_messages=(*state.recent_messages, user_message),
        )
        decision = self._decide(state_with_user, user_message)
        trace_event("route_started", route=decision.route)

        if decision.route is CoordinatorRoute.CHAT:
            routed_state, contents = self._chat(state_with_user, user_message)
        elif decision.route is CoordinatorRoute.START_WRITING_TASK:
            routed_state, contents = self._start_task(state_with_user, decision)
        elif decision.route is CoordinatorRoute.APPROVE_TASK:
            routed_state, contents = self._approve_task(state_with_user, user_message)
        elif decision.route is CoordinatorRoute.REVISE_TASK:
            routed_state, contents = self._revise_task(state_with_user, decision)
        else:
            raise ConversationServiceError("Coordinator returned an unsupported route")

        assistant_messages = tuple(
            self._message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=content,
            )
            for content in contents
        )
        completed_state = replace(
            routed_state,
            recent_messages=(
                *routed_state.recent_messages,
                *assistant_messages,
            )[-self._max_recent_messages :],
        )
        self._save_state(completed_state)
        return assistant_messages

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
        return state, (response,)

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
            status=WritingTaskStatus.AWAITING_USER_EVALUATION,
            created_at=task.created_at,
            updated_at=self._timestamp(),
            working_draft=result.working_draft,
            critic_report=result.critic_report,
        )
        routed_state = replace(
            state,
            status=ConversationStatus.AWAITING_USER_EVALUATION,
            active_task=active_task,
        )
        return routed_state, self._writing_messages(result)

    def _approve_task(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> tuple[ConversationState, tuple[str, ...]]:
        set_trace_stage("approval")
        trace_event("approval_started", stage="approval")
        task = self._require_awaiting_task(state, require_draft=True)
        approved_task = replace(
            task,
            status=WritingTaskStatus.APPROVED,
            user_evaluation=user_message.content,
            updated_at=self._timestamp(),
        )
        routed_state = replace(
            state,
            status=ConversationStatus.CHATTING,
            active_task=approved_task,
        )
        acknowledgement = self._talk(routed_state, user_message)
        set_trace_stage("approval")
        trace_event("approval_completed", stage="approval", outcome="completed")
        return routed_state, (acknowledgement,)

    def _revise_task(
        self,
        state: ConversationState,
        decision: CoordinatorDecision,
    ) -> tuple[ConversationState, tuple[str, ...]]:
        set_trace_stage("revision_workflow")
        trace_event("revision_workflow_started", stage="revision_workflow")
        task = self._require_awaiting_task(state, require_draft=True)
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
                user_evaluation=None,
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
            status=WritingTaskStatus.AWAITING_USER_EVALUATION,
            working_draft=result.working_draft,
            critic_report=result.critic_report,
            user_evaluation=None,
            updated_at=self._timestamp(),
        )
        routed_state = replace(
            state,
            status=ConversationStatus.AWAITING_USER_EVALUATION,
            active_task=active_task,
        )
        return routed_state, self._writing_messages(result)

    def _require_awaiting_task(
        self,
        state: ConversationState,
        *,
        require_draft: bool,
    ) -> WritingTask:
        task = state.active_task
        if (
            state.status is not ConversationStatus.AWAITING_USER_EVALUATION
            or task is None
            or task.status is not WritingTaskStatus.AWAITING_USER_EVALUATION
            or task.critic_report is None
            or (require_draft and task.working_draft is None)
        ):
            raise ConversationServiceError("No writing task is awaiting user evaluation")
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
            f"Writer output:\n{result.writer_output}",
            format_critic_report(result.critic_report),
            format_working_draft(result),
            request_user_evaluation(),
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
