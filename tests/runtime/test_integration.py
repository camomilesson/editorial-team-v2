from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.domain.conversation import ConversationState, Message
from editorial_team.domain.editorial import (
    CriticReport,
    CriticVerdict,
    EditorialResult,
    WritingTask,
)
from editorial_team.domain.routing import CoordinatorDecision, CoordinatorRoute
from editorial_team.runtime import RuntimeJobSource, RuntimeQueue, RuntimeQueueError

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


class ContentCoordinator:
    def decide(
        self,
        state: ConversationState,
        user_message: Message,
    ) -> CoordinatorDecision:
        del state
        if user_message.content.startswith("start "):
            return CoordinatorDecision(
                CoordinatorRoute.START_WRITING_TASK,
                1.0,
                task_input=user_message.content.removeprefix("start "),
            )
        return CoordinatorDecision(
            CoordinatorRoute.REVISE_TASK,
            1.0,
            revision_instructions=user_message.content,
        )


class UnusedTalker:
    def respond(self, state: ConversationState, user_message: Message) -> str:
        del state, user_message
        raise AssertionError("Talker must not be called")


class PassingWorkflow:
    def execute(self, task: WritingTask) -> EditorialResult:
        instruction = task.brief.instructions[-1] if task.brief.instructions else ""
        draft = f"{task.brief.original_request}|{instruction}"
        report = CriticReport(CriticVerdict.PASS, "Ready.")
        return EditorialResult(draft, report, draft, False)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"id-{self.value}"


def test_real_queue_preserves_fifo_failure_isolation_and_conversation_scope() -> None:
    async def scenario() -> None:
        store = InMemoryConversationStateStore()
        service = ConversationService(
            coordinator=ContentCoordinator(),
            talker=UnusedTalker(),
            workflow=PassingWorkflow(),
            store=store,
            identifier_generator=Ids(),
            clock=lambda: NOW,
            max_recent_messages=50,
        )
        queue = RuntimeQueue()
        await queue.start()

        async def turn(conversation_id: str, text: str):
            return await asyncio.to_thread(
                service.process_message,
                conversation_id,
                text,
            )

        await queue.submit(
            source=RuntimeJobSource.TELEGRAM,
            correlation_id="multi-user-a-start",
            operation=lambda: turn("conversation-a", "start A draft"),
        )
        a_before = store.load("conversation-a")
        assert a_before is not None and a_before.active_task is not None
        a_task_id = a_before.active_task.id

        b_revision = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="multi-user-b-revise",
                operation=lambda: turn("conversation-b", "make it shorter"),
            )
        )
        a_revision = asyncio.create_task(
            queue.submit(
                source=RuntimeJobSource.TELEGRAM,
                correlation_id="multi-user-a-revise",
                operation=lambda: turn("conversation-a", "make it warmer"),
            )
        )
        results = await asyncio.gather(
            b_revision,
            a_revision,
            return_exceptions=True,
        )

        assert isinstance(results[0], RuntimeQueueError)
        assert not isinstance(results[1], Exception)
        assert store.load("conversation-b") is None
        a_after = store.load("conversation-a")
        assert a_after is not None and a_after.active_task is not None
        assert a_after.active_task.id == a_task_id
        assert a_after.active_task.brief.instructions == ("make it warmer",)

        await queue.submit(
            source=RuntimeJobSource.TELEGRAM,
            correlation_id="multi-user-b-start",
            operation=lambda: turn("conversation-b", "start B draft"),
        )
        b_after = store.load("conversation-b")
        final_a = store.load("conversation-a")
        assert b_after is not None and b_after.active_task is not None
        assert final_a is not None and final_a.active_task is not None
        assert b_after.active_task.id != final_a.active_task.id
        assert final_a.active_task.id == a_task_id
        assert final_a.active_task.working_draft == "A draft|make it warmer"

        await queue.close()

    asyncio.run(scenario())
