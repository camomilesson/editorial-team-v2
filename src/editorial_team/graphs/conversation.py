"""Partially implemented parent conversation routing graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from editorial_team.contracts.common import require_non_blank
from editorial_team.contracts.identity import validate_identifier
from editorial_team.domain.conversation import ConversationState, Message, MessageRole
from editorial_team.domain.editorial import (
    CriticReport,
    EditorialResult,
    WritingBrief,
    WritingTask,
    WritingTaskStatus,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.graphs.editorial import build_editorial_subgraph
from editorial_team.graphs.state import (
    EditorialGraphStateV1,
    validate_graph_state_version,
)

if TYPE_CHECKING:
    from editorial_team.agents.protocols import Critic, Editor, Writer
    from editorial_team.conversation.protocols import (
        ConversationStateStore,
        Coordinator,
        Talker,
        WritingWorkflowRunner,
    )
    from editorial_team.conversation.service import ConversationService

IdentifierGenerator = Callable[[], str]
UtcClock = Callable[[], datetime]

ParentRoute = Literal["chat", "start_writing_task", "revise_task"]


class _WorkflowGraphAdapter:
    """Expose an existing workflow runner through the editorial node contract."""

    def __init__(self, workflow: WritingWorkflowRunner) -> None:
        self._workflow = workflow

    def invoke(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        task = state.get("writing_task")
        if not isinstance(task, WritingTask):
            raise ValueError("Invalid writing task")
        result = self._workflow.execute(task)
        if not isinstance(result, EditorialResult):
            return {"editorial_result": result}  # type: ignore[typeddict-item]
        return {
            "writer_output": result.writer_output,
            "critic_report": result.critic_report,
            "working_draft": result.working_draft,
            "editorial_result": result,
        }


class _ParentRoutingNodes:
    """Node adapters around the current ConversationService behavior."""

    def __init__(
        self,
        service: ConversationService,
        error_type: type[Exception],
        editorial_graph: Any,
        max_recent_messages: int,
    ) -> None:
        self._service = service
        self._error_type = error_type
        self._editorial_graph = editorial_graph
        self._max_recent_messages = max_recent_messages

    def validate_and_prepare_turn(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Validate input, load prior state, and create the exact user message."""

        validate_graph_state_version(state)
        conversation_id = state.get("conversation_id")
        text = state.get("input_text")
        try:
            conversation_id = validate_identifier(conversation_id, "conversation_id")  # type: ignore[arg-type]
            require_non_blank(text, "text")  # type: ignore[arg-type]
        except ValueError:
            raise self._error_type("Invalid conversation input") from None

        prior = self._service._load_state(conversation_id)
        user_message = self._service._message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=text,  # type: ignore[arg-type]
        )
        return {
            "conversation_id": conversation_id,
            "prior_conversation": prior,
            "user_message": user_message,
        }

    def coordinator(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Return the same validated Coordinator decision as ConversationService."""

        prepared, user_message = self._prepared_turn(state)
        decision = self._service._decide(prepared, user_message)
        from editorial_team.tracing import trace_event

        trace_event("route_started", route=decision.route)
        return {"decision": decision}

    def talker(self, state: EditorialGraphStateV1) -> EditorialGraphStateV1:
        """Return the same validated plain-text Talker response."""

        prepared, user_message = self._prepared_turn(state)
        return {"talker_response": self._service._talk(prepared, user_message)}

    def prepare_new_task(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Create the same initial task as ConversationService."""

        from editorial_team.tracing import set_trace_stage, trace_event

        set_trace_stage("writing_workflow")
        trace_event("writing_workflow_started", stage="writing_workflow")
        decision = self._require_decision(state)
        if decision.task_input is None:
            raise self._error_type("Writing task input is invalid")
        conversation_id = state.get("conversation_id")
        timestamp = self._service._timestamp()
        task = WritingTask(
            id=self._service._identifier("task"),
            conversation_id=conversation_id,  # type: ignore[arg-type]
            brief=WritingBrief(decision.task_input),
            status=WritingTaskStatus.CREATED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return {"writing_task": task}

    def prepare_revision(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Prepare the latest task with one appended revision instruction."""

        from editorial_team.tracing import set_trace_stage, trace_event

        set_trace_stage("revision_workflow")
        trace_event("revision_workflow_started", stage="revision_workflow")
        prepared, _ = self._prepared_turn(state)
        task = self._service._require_latest_task(prepared)
        decision = self._require_decision(state)
        if decision.revision_instructions is None:
            raise self._error_type("Revision instructions are invalid")
        try:
            brief = replace(
                task.brief,
                instructions=(
                    *task.brief.instructions,
                    decision.revision_instructions,
                ),
            )
            workflow_task = replace(task, brief=brief)
        except (TypeError, ValueError):
            raise self._error_type("Revision task is invalid") from None
        return {"writing_task": workflow_task}

    def editorial_subgraph(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Run the existing editorial subgraph behind the service error boundary."""

        from editorial_team.tracing import (
            current_trace_stage,
            error_category,
            set_trace_stage,
            trace_event,
        )

        try:
            output = self._editorial_graph.invoke(state)
        except Exception as exc:
            trace_event(
                "writing_workflow_failed",
                stage=current_trace_stage(),
                outcome="failed",
                error_category=error_category(exc),
            )
            raise self._error_type("Writing workflow failed") from None
        set_trace_stage("writing_workflow")
        result = output.get("editorial_result")
        if not isinstance(result, EditorialResult):
            trace_event(
                "writing_workflow_failed",
                stage="writing_workflow",
                outcome="failed",
                error_category="schema_validation_failure",
            )
            raise self._error_type("Writing workflow returned an invalid result")
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
            raise self._error_type("Writing workflow returned an invalid result") from None
        return {
            "writer_output": result.writer_output,
            "critic_report": result.critic_report,
            "working_draft": result.working_draft,
            "editorial_result": result,
        }

    def finalize_task(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Map a successful editorial result to the canonical active task."""

        from editorial_team.tracing import set_trace_stage, trace_event

        decision = self._require_decision(state)
        task = state.get("writing_task")
        result = state.get("editorial_result")
        if not isinstance(task, WritingTask) or not isinstance(result, EditorialResult):
            raise self._error_type("Writing workflow returned an invalid result")

        if decision.route is CoordinatorRoute.START_WRITING_TASK:
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
                status=self._service._completed_status(result),
                created_at=task.created_at,
                updated_at=self._service._timestamp(),
                working_draft=result.working_draft,
                critic_report=result.critic_report,
            )
        elif decision.route is CoordinatorRoute.REVISE_TASK:
            set_trace_stage("revision_workflow")
            trace_event(
                "revision_workflow_completed",
                stage="revision_workflow",
                outcome="completed",
                critic_verdict=result.critic_report.verdict,
                revision_applied=result.revision_applied,
            )
            active_task = replace(
                task,
                status=self._service._completed_status(result),
                working_draft=result.working_draft,
                critic_report=result.critic_report,
                updated_at=self._service._timestamp(),
            )
        else:
            raise self._error_type("Coordinator returned an unsupported route")

        prepared, _ = self._prepared_turn(state)
        return {"routed_conversation": replace(prepared, active_task=active_task)}

    def finalize_turn(
        self,
        state: EditorialGraphStateV1,
    ) -> EditorialGraphStateV1:
        """Construct exact assistant messages and completed conversation state."""

        from editorial_team.conversation.formatting import format_talker_message

        decision = self._require_decision(state)
        if decision.route is CoordinatorRoute.CHAT:
            routed_state, _ = self._prepared_turn(state)
            response = state.get("talker_response")
            if not isinstance(response, str):
                raise self._error_type("Talker returned an invalid response")
            contents = (format_talker_message(response),)
        else:
            routed_state = state.get("routed_conversation")
            result = state.get("editorial_result")
            if not isinstance(routed_state, ConversationState) or not isinstance(
                result, EditorialResult
            ):
                raise self._error_type("Writing workflow returned an invalid result")
            contents = self._service._writing_messages(result)

        conversation_id = state.get("conversation_id")
        assistant_messages = tuple(
            self._service._message(
                conversation_id=conversation_id,  # type: ignore[arg-type]
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
        return {
            "assistant_contents": contents,
            "assistant_messages": assistant_messages,
            "completed_conversation": completed_state,
        }

    def _require_decision(
        self,
        state: EditorialGraphStateV1,
    ) -> CoordinatorDecision:
        decision = state.get("decision")
        if not isinstance(decision, CoordinatorDecision):
            raise self._error_type("Coordinator returned an invalid decision")
        return decision

    def _prepared_turn(
        self,
        state: EditorialGraphStateV1,
    ) -> tuple[ConversationState, Message]:
        prior = state.get("prior_conversation")
        user_message = state.get("user_message")
        if not isinstance(prior, ConversationState) or not isinstance(
            user_message, Message
        ):
            raise self._error_type("Invalid conversation input")
        return (
            replace(
                prior,
                recent_messages=(*prior.recent_messages, user_message),
            ),
            user_message,
        )


def _route_coordinator_decision(state: EditorialGraphStateV1) -> ParentRoute:
    """Describe the future parent branch using a validated domain decision."""

    decision = state.get("decision")
    if decision is None:
        raise ValueError("Parent graph state has no Coordinator decision")
    return decision.route.value


def build_parent_graph(
    *,
    coordinator: Coordinator,
    talker: Talker,
    store: ConversationStateStore,
    identifier_generator: IdentifierGenerator,
    clock: UtcClock,
    max_recent_messages: int,
    writer: Writer | None = None,
    critic: Critic | None = None,
    editor: Editor | None = None,
    workflow: WritingWorkflowRunner | None = None,
) -> StateGraph[EditorialGraphStateV1]:
    """Build the complete but disconnected parent routing graph.

    The graph mirrors ConversationService but is not connected to application
    composition, repositories, transports, or any production request path.
    """

    from editorial_team.conversation.service import (
        ConversationService,
        ConversationServiceError,
    )

    if workflow is not None:
        if any(participant is not None for participant in (writer, critic, editor)):
            raise ValueError("Provide either workflow or editorial participants")
        editorial_graph: Any = _WorkflowGraphAdapter(workflow)
    else:
        if writer is None or critic is None or editor is None:
            raise ValueError("Editorial participants are required")
        editorial_graph = build_editorial_subgraph(
            writer=writer,
            critic=critic,
            editor=editor,
        ).compile()

    service = object.__new__(ConversationService)
    service._coordinator = coordinator
    service._talker = talker
    service._workflow = workflow
    service._store = store
    service._identifier_generator = identifier_generator
    service._clock = clock
    service._max_recent_messages = max_recent_messages
    nodes = _ParentRoutingNodes(
        service,
        ConversationServiceError,
        editorial_graph,
        max_recent_messages,
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
            CoordinatorRoute.CHAT.value: "talker",
            CoordinatorRoute.START_WRITING_TASK.value: "prepare_new_task",
            CoordinatorRoute.REVISE_TASK.value: "prepare_revision",
        },
    )
    graph.add_edge("talker", "finalize_turn")
    graph.add_edge("prepare_new_task", "editorial_subgraph")
    graph.add_edge("prepare_revision", "editorial_subgraph")
    graph.add_edge("editorial_subgraph", "finalize_task")
    graph.add_edge("finalize_task", "finalize_turn")
    graph.add_edge("finalize_turn", END)
    return graph
