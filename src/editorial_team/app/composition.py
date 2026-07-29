"""Composition root for the local live Telegram product."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from telegram.ext import Application

from editorial_team.agents import (
    LlmCoordinator,
    LlmCritic,
    LlmEditor,
    LlmTalker,
    LlmWriter,
)
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.gemini import create_gemini_client_from_env
from editorial_team.interfaces.telegram import TelegramAdapter, build_telegram_application
from editorial_team.models import ModelClient
from editorial_team.runtime import DEFAULT_RUNTIME_QUEUE_CAPACITY, RuntimeQueue
from editorial_team.workflows import WritingWorkflow

RECENT_MESSAGE_LIMIT = 50


class LiveConfigurationError(RuntimeError):
    """Required live configuration is absent or invalid."""


@dataclass(frozen=True)
class LiveApplication:
    """Composed live objects retained for startup and inspection."""

    telegram: Application
    service: ConversationService
    store: InMemoryConversationStateStore
    adapter: TelegramAdapter
    runtime_queue: RuntimeQueue
    model_name: str


def build_conversation_service(
    model: ModelClient,
) -> tuple[ConversationService, InMemoryConversationStateStore]:
    """Wire the real agents around one shared provider-neutral model client."""

    store = InMemoryConversationStateStore()
    workflow = WritingWorkflow(
        writer=LlmWriter(model),
        critic=LlmCritic(model),
        editor=LlmEditor(model),
    )
    service = ConversationService(
        coordinator=LlmCoordinator(model),
        talker=LlmTalker(model),
        workflow=workflow,
        store=store,
        identifier_generator=lambda: uuid4().hex,
        clock=lambda: datetime.now(UTC),
        max_recent_messages=RECENT_MESSAGE_LIMIT,
    )
    return service, store


def build_live_application_from_env() -> LiveApplication:
    """Validate process configuration and compose the polling application."""

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise LiveConfigurationError("Required Telegram configuration is missing")

    try:
        model = create_gemini_client_from_env()
    except Exception:
        raise LiveConfigurationError("Required model configuration is missing or invalid") from None

    service, store = build_conversation_service(model)
    runtime_queue = RuntimeQueue(DEFAULT_RUNTIME_QUEUE_CAPACITY)
    adapter = TelegramAdapter(service, runtime_queue)
    try:
        telegram = build_telegram_application(token=token, adapter=adapter)
    except Exception:
        raise LiveConfigurationError("Telegram configuration is invalid") from None
    return LiveApplication(
        telegram=telegram,
        service=service,
        store=store,
        adapter=adapter,
        runtime_queue=runtime_queue,
        model_name=model.model,
    )
