"""Composition root for live Editorial Team interfaces."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram.ext import Application

from editorial_team.agents import (
    LlmAdminAgent,
    LlmCoordinator,
    LlmCritic,
    LlmEditor,
    LlmTalker,
    LlmWriter,
    ToolCallingCoordinator,
)
from editorial_team.app.artifact_config import (
    ArtifactConfigurationError,
    load_artifact_configuration,
)
from editorial_team.app.checkpoint_config import (
    CheckpointConfigurationError,
    load_checkpoint_configuration,
)
from editorial_team.app.heartbeat_config import (
    HeartbeatConfigurationError,
    load_heartbeat_configuration,
)
from editorial_team.app.retrieval_config import (
    RetrievalConfiguration,
    RetrievalConfigurationError,
    load_retrieval_configuration,
)
from editorial_team.app.telegram_config import (
    TelegramConfigurationError,
    load_telegram_configuration,
)
from editorial_team.artifacts import HybridRetriever, ParagraphChunker, SQLiteArtifactStore
from editorial_team.artifacts.embeddings import SentenceTransformerEmbeddingModel
from editorial_team.artifacts.reranking import CrossEncoderReranker
from editorial_team.conversation import ConversationService
from editorial_team.gemini import (
    create_gemini_chat_model_from_env,
    create_gemini_client_from_env,
)
from editorial_team.graphs import build_parent_graph, create_sqlite_checkpointer
from editorial_team.interfaces.admin import TelegramMaintainerNotifier
from editorial_team.interfaces.telegram import TelegramAdapter, build_telegram_application
from editorial_team.mlflow_tracing import initialize_mlflow_tracing
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

RECENT_MESSAGE_LIMIT = 50


class LiveConfigurationError(RuntimeError):
    """Required live configuration is absent or invalid."""


@dataclass(frozen=True)
class LiveApplication:
    """Composed live objects retained for startup and inspection."""

    telegram: Application
    service: ConversationService
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


def build_conversation_service(
    model: ModelClient,
    checkpoint_path: Path,
    *,
    artifact_path: Path | None = None,
    busy_timeout_seconds: float = 5.0,
    coordinator_chat_model: object | None = None,
    retrieval_configuration: RetrievalConfiguration | None = None,
    user_timezone: str = "Europe/Madrid",
    clock: Callable[[], datetime] | None = None,
    protected_output_markers: tuple[str, ...] = (),
) -> ConversationService:
    """Wire the real agents around one shared provider-neutral model client."""

    coordinator = LlmCoordinator(model)
    talker = LlmTalker(model)
    writer = LlmWriter(model)
    critic = LlmCritic(model)
    editor = LlmEditor(model)

    def identifier_generator() -> str:
        return uuid4().hex

    def system_clock() -> datetime:
        return datetime.now(UTC)

    evaluation_or_system_clock = clock or system_clock

    artifact_store = SQLiteArtifactStore(
        artifact_path or checkpoint_path.with_name("editorial_artifacts.db"),
        chunker=ParagraphChunker(),
    )
    artifact_store.initialize()
    try:
        retriever = None
        tool_coordinator = None
        if coordinator_chat_model is not None:
            configuration = retrieval_configuration or RetrievalConfiguration()
            retriever = HybridRetriever(
                store=artifact_store,
                embeddings=SentenceTransformerEmbeddingModel(configuration.embedding_model),
                reranker=CrossEncoderReranker(configuration.reranker_model),
                dense_depth=configuration.dense_depth,
                bm25_depth=configuration.bm25_depth,
                rrf_k=configuration.rrf_k,
                fused_depth=configuration.fused_depth,
                rerank_depth=configuration.rerank_depth,
            )
            tool_coordinator = ToolCallingCoordinator(coordinator_chat_model)
    except BaseException:
        artifact_store.close()
        raise
    try:
        checkpointer, close_checkpointer = create_sqlite_checkpointer(
            checkpoint_path,
            busy_timeout_seconds=busy_timeout_seconds,
        )
    except BaseException:
        artifact_store.close()
        raise
    try:
        graph_runner = build_parent_graph(
            coordinator=coordinator,
            talker=talker,
            writer=writer,
            critic=critic,
            editor=editor,
            identifier_generator=identifier_generator,
            clock=evaluation_or_system_clock,
            max_recent_messages=RECENT_MESSAGE_LIMIT,
            artifact_store=artifact_store,
            tool_coordinator=tool_coordinator,
            retriever=retriever,
            user_timezone=user_timezone,
        ).compile(checkpointer=checkpointer)
    except BaseException:
        close_checkpointer()
        artifact_store.close()
        raise

    def close_resources() -> None:
        failure: BaseException | None = None
        try:
            close_checkpointer()
        except BaseException as exc:
            failure = exc
        try:
            artifact_store.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure

    return ConversationService(
        graph_runner=graph_runner,
        close_checkpointer=close_resources,
        protected_output_markers=protected_output_markers,
    )


def build_live_application_from_env() -> LiveApplication:
    """Validate process configuration and compose the polling application."""

    initialize_mlflow_tracing()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise LiveConfigurationError("Required Telegram configuration is missing")

    try:
        artifact_configuration = load_artifact_configuration()
    except ArtifactConfigurationError:
        raise LiveConfigurationError("Artifact configuration is invalid") from None
    try:
        heartbeat_configuration = load_heartbeat_configuration()
    except HeartbeatConfigurationError:
        raise LiveConfigurationError("Heartbeat configuration is invalid") from None
    try:
        telegram_configuration = load_telegram_configuration()
    except TelegramConfigurationError:
        raise LiveConfigurationError("Telegram configuration is invalid") from None
    try:
        checkpoint_configuration = load_checkpoint_configuration()
    except CheckpointConfigurationError:
        raise LiveConfigurationError("Checkpoint configuration is invalid") from None
    try:
        retrieval_configuration = load_retrieval_configuration()
    except RetrievalConfigurationError:
        raise LiveConfigurationError("Retrieval configuration is invalid") from None
    user_timezone = os.getenv("EDITORIAL_USER_TIMEZONE", "Europe/Madrid").strip()
    try:
        ZoneInfo(user_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        raise LiveConfigurationError("User timezone configuration is invalid") from None

    try:
        model = create_gemini_client_from_env()
        coordinator_chat_model = create_gemini_chat_model_from_env()
    except Exception:
        raise LiveConfigurationError("Required model configuration is missing or invalid") from None

    try:
        service = build_conversation_service(
            model,
            checkpoint_configuration.database_path,
            artifact_path=artifact_configuration.database_path,
            busy_timeout_seconds=checkpoint_configuration.busy_timeout_seconds,
            coordinator_chat_model=coordinator_chat_model,
            retrieval_configuration=retrieval_configuration,
            user_timezone=user_timezone,
        )
    except Exception:
        raise LiveConfigurationError("Checkpoint database could not be initialized") from None
    runtime_queue = RuntimeQueue(DEFAULT_RUNTIME_QUEUE_CAPACITY)
    try:
        adapter = TelegramAdapter(
            service,
            runtime_queue,
            allowed_chat_ids=telegram_configuration.allowed_chat_ids,
        )
        telegram = build_telegram_application(token=token, adapter=adapter)
    except Exception:
        service.close()
        raise LiveConfigurationError("Telegram configuration is invalid") from None
    heartbeat: HeartbeatComponents | None = None
    try:
        if heartbeat_configuration.enabled:
            if heartbeat_configuration.maintainer_chat_id is None:
                raise LiveConfigurationError("Heartbeat configuration is invalid")
            heartbeat_store = SQLiteHeartbeatResultStore(heartbeat_configuration.database_path)
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
    except BaseException:
        service.close()
        raise
    return LiveApplication(
        telegram=telegram,
        service=service,
        adapter=adapter,
        runtime_queue=runtime_queue,
        model_name=model.model,
        heartbeat=heartbeat,
    )
