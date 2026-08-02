"""Composition root for live Editorial Team interfaces."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from telegram.ext import Application

from editorial_team.agents import (
    LlmAdminAgent,
    LlmCoordinator,
    LlmCritic,
    LlmEditor,
    LlmTalker,
    LlmWriter,
)
from editorial_team.app.external_config import (
    ExternalApiConfiguration,
    ExternalApiConfigurationError,
    load_external_api_configuration,
)
from editorial_team.app.heartbeat_config import (
    HeartbeatConfigurationError,
    load_heartbeat_configuration,
)
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.gemini import create_gemini_client_from_env
from editorial_team.graphs import build_parent_graph, create_in_memory_checkpointer
from editorial_team.interfaces.admin import TelegramMaintainerNotifier
from editorial_team.interfaces.external_http import (
    ExternalBriefHttpAdapter,
    ExternalBriefHttpServer,
)
from editorial_team.interfaces.telegram import TelegramAdapter, build_telegram_application
from editorial_team.models import ModelClient
from editorial_team.operations import (
    AdminPolicy,
    HeartbeatEvaluationService,
    HeartbeatRunner,
    HeartbeatScheduler,
    OperationalSnapshotCollector,
    SQLiteHeartbeatResultStore,
)
from editorial_team.runtime import DEFAULT_RUNTIME_QUEUE_CAPACITY, RuntimeQueue
from editorial_team.tracing import trace_runtime_event
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
    heartbeat: HeartbeatComponents | None


@dataclass(frozen=True)
class HeartbeatComponents:
    """Exactly one composed instance of every optional heartbeat component."""

    store: SQLiteHeartbeatResultStore
    admin_agent: LlmAdminAgent
    evaluation_service: HeartbeatEvaluationService
    collector: OperationalSnapshotCollector
    notifier: TelegramMaintainerNotifier
    runner: HeartbeatRunner
    scheduler: HeartbeatScheduler


@dataclass(frozen=True)
class ExternalApiApplication:
    """Composed dependencies for the external brief server."""

    service: ConversationService
    store: InMemoryConversationStateStore
    adapter: ExternalBriefHttpAdapter
    runtime_queue: RuntimeQueue
    model_name: str


@dataclass(frozen=True)
class CombinedLiveApplication:
    """Telegram, external HTTP, and heartbeat sharing one live dependency graph."""

    live: LiveApplication
    external_adapter: ExternalBriefHttpAdapter
    external_configuration: ExternalApiConfiguration
    lifecycle: CombinedRuntimeLifecycle

    @property
    def runtime_queue(self) -> RuntimeQueue:
        """Expose the single shared queue for inspection."""

        return self.live.runtime_queue


class CombinedRuntimeLifecycle:
    """Own the HTTP server around the existing Telegram runtime lifecycle."""

    def __init__(
        self,
        *,
        live: LiveApplication,
        external_adapter: ExternalBriefHttpAdapter,
        configuration: ExternalApiConfiguration,
        server_factory: type[ExternalBriefHttpServer] = ExternalBriefHttpServer,
    ) -> None:
        self._live = live
        self._external_adapter = external_adapter
        self._configuration = configuration
        self._server_factory = server_factory
        self._server: ExternalBriefHttpServer | None = None
        self._server_task: asyncio.Task[None] | None = None

    async def start(self, application: Application) -> None:
        """Bind HTTP, then start the one shared queue and heartbeat lifecycle."""

        loop = asyncio.get_running_loop()
        try:
            server = self._server_factory(
                (self._configuration.host, self._configuration.port),
                adapter=self._external_adapter,
                loop=loop,
            )
        except Exception:
            raise LiveConfigurationError("External HTTP server configuration is invalid") from None
        self._server = server
        try:
            await self._live.adapter.start_runtime(application)
            self._server_task = asyncio.create_task(
                asyncio.to_thread(server.serve_forever),
                name="external-http-server",
            )
        except Exception:
            await asyncio.to_thread(server.server_close)
            await self._live.adapter.close_runtime(application)
            raise LiveConfigurationError("Combined runtime could not start") from None
        trace_runtime_event("combined_runtime_started", correlation_id="combined-runtime")
        trace_runtime_event("external_server_started", correlation_id="external-server")

    async def close(self, application: Application) -> None:
        """Stop HTTP acceptance and server before closing heartbeat and queue."""

        server = self._server
        if server is not None:
            self._external_adapter.stop_accepting()
            await asyncio.to_thread(server.shutdown)
            if self._server_task is not None:
                await self._server_task
            await asyncio.to_thread(server.server_close)
            trace_runtime_event("external_server_stopped", correlation_id="external-server")
        await self._live.adapter.close_runtime(application)
        trace_runtime_event("combined_runtime_stopped", correlation_id="combined-runtime")


def build_conversation_service(
    model: ModelClient,
) -> tuple[ConversationService, InMemoryConversationStateStore]:
    """Wire the real agents around one shared provider-neutral model client."""

    store = InMemoryConversationStateStore()
    coordinator = LlmCoordinator(model)
    talker = LlmTalker(model)
    writer = LlmWriter(model)
    critic = LlmCritic(model)
    editor = LlmEditor(model)

    def identifier_generator() -> str:
        return uuid4().hex

    def clock() -> datetime:
        return datetime.now(UTC)

    workflow = WritingWorkflow(
        writer=writer,
        critic=critic,
        editor=editor,
    )
    checkpointer = create_in_memory_checkpointer()
    graph_runner = build_parent_graph(
        coordinator=coordinator,
        talker=talker,
        writer=writer,
        critic=critic,
        editor=editor,
        store=store,
        identifier_generator=identifier_generator,
        clock=clock,
        max_recent_messages=RECENT_MESSAGE_LIMIT,
    ).compile(checkpointer=checkpointer)
    service = ConversationService(
        coordinator=coordinator,
        talker=talker,
        workflow=workflow,
        store=store,
        identifier_generator=identifier_generator,
        clock=clock,
        max_recent_messages=RECENT_MESSAGE_LIMIT,
        graph_runner=graph_runner,
        graph_checkpointer=checkpointer,
    )
    return service, store


def build_live_application_from_env() -> LiveApplication:
    """Validate process configuration and compose the polling application."""

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise LiveConfigurationError("Required Telegram configuration is missing")

    try:
        heartbeat_configuration = load_heartbeat_configuration()
    except HeartbeatConfigurationError:
        raise LiveConfigurationError("Heartbeat configuration is invalid") from None

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
    heartbeat: HeartbeatComponents | None = None
    if heartbeat_configuration.enabled:
        if heartbeat_configuration.maintainer_chat_id is None:
            raise LiveConfigurationError("Heartbeat configuration is invalid")
        heartbeat_store = SQLiteHeartbeatResultStore(
            heartbeat_configuration.database_path
        )
        admin_agent = LlmAdminAgent(model)
        evaluation_service = HeartbeatEvaluationService(
            admin_agent=admin_agent,
            store=heartbeat_store,
            policy=AdminPolicy(),
            identifier_generator=lambda: f"heartbeat-result-{uuid4().hex}",
        )
        collector = OperationalSnapshotCollector(runtime_queue)
        notifier = TelegramMaintainerNotifier(
            telegram.bot,
            heartbeat_configuration.maintainer_chat_id,
        )
        runner = HeartbeatRunner(
            runtime_queue=runtime_queue,
            collector=collector,
            evaluation_service=evaluation_service,
            store=heartbeat_store,
            notifier=notifier,
        )
        scheduler = HeartbeatScheduler(
            runner,
            interval_seconds=heartbeat_configuration.interval_seconds,
        )
        adapter.configure_heartbeat(store=heartbeat_store, scheduler=scheduler)
        heartbeat = HeartbeatComponents(
            store=heartbeat_store,
            admin_agent=admin_agent,
            evaluation_service=evaluation_service,
            collector=collector,
            notifier=notifier,
            runner=runner,
            scheduler=scheduler,
        )
    return LiveApplication(
        telegram=telegram,
        service=service,
        store=store,
        adapter=adapter,
        runtime_queue=runtime_queue,
        model_name=model.model,
        heartbeat=heartbeat,
    )


def build_external_api_application(token: str) -> ExternalApiApplication:
    """Compose the external adapter around one shared model, service, and queue."""

    if not isinstance(token, str) or not token.strip():
        raise LiveConfigurationError("Required external API configuration is missing")
    try:
        model = create_gemini_client_from_env()
    except Exception:
        raise LiveConfigurationError("Required model configuration is missing or invalid") from None
    service, store = build_conversation_service(model)
    runtime_queue = RuntimeQueue(DEFAULT_RUNTIME_QUEUE_CAPACITY)
    adapter = ExternalBriefHttpAdapter(
        token=token,
        service=service,
        runtime_queue=runtime_queue,
    )
    return ExternalApiApplication(
        service=service,
        store=store,
        adapter=adapter,
        runtime_queue=runtime_queue,
        model_name=model.model,
    )


def build_combined_live_application_from_env() -> CombinedLiveApplication:
    """Compose Telegram, HTTP, and heartbeat over one model, service, and queue."""

    try:
        configuration = load_external_api_configuration()
    except ExternalApiConfigurationError:
        raise LiveConfigurationError("Required external API configuration is missing") from None
    live = build_live_application_from_env()
    external_adapter = ExternalBriefHttpAdapter(
        token=configuration.token,
        service=live.service,
        runtime_queue=live.runtime_queue,
    )
    lifecycle = CombinedRuntimeLifecycle(
        live=live,
        external_adapter=external_adapter,
        configuration=configuration,
    )
    live.telegram.post_init = lifecycle.start
    live.telegram.post_shutdown = lifecycle.close
    return CombinedLiveApplication(
        live=live,
        external_adapter=external_adapter,
        external_configuration=configuration,
        lifecycle=lifecycle,
    )
