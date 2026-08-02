from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from telegram.ext import Application

from editorial_team.app import composition
from editorial_team.app.composition import (
    RECENT_MESSAGE_LIMIT,
    CombinedLiveApplication,
    CombinedRuntimeLifecycle,
    ExternalApiApplication,
    LiveConfigurationError,
    build_combined_live_application_from_env,
    build_conversation_service,
    build_external_api_application,
    build_live_application_from_env,
)
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.models import FakeModelClient, ModelResponse
from editorial_team.operations import (
    HeartbeatEvaluationService,
    HeartbeatRunner,
    HeartbeatScheduler,
    OperationalSnapshotCollector,
    SQLiteHeartbeatResultStore,
)
from editorial_team.runtime import DEFAULT_RUNTIME_QUEUE_CAPACITY, RuntimeJobSource, RuntimeQueue


class NamedFakeModel(FakeModelClient):
    model = "safe-test-model"


def test_build_conversation_service_wires_real_components_with_in_memory_store() -> None:
    model = NamedFakeModel([])

    service, store = build_conversation_service(model)

    assert isinstance(service, ConversationService)
    assert isinstance(store, InMemoryConversationStateStore)
    assert service._store is store
    assert service._max_recent_messages == RECENT_MESSAGE_LIMIT
    assert service._coordinator._model is model
    assert service._talker._model is model
    assert service._workflow._writer._model is model
    assert service._workflow._critic._model is model
    assert service._workflow._editor._model is model
    assert isinstance(service._graph_checkpointer, InMemorySaver)
    assert service._graph_runner.checkpointer is service._graph_checkpointer


def test_composed_conversation_service_executes_the_compiled_parent_graph() -> None:
    model = NamedFakeModel(
        [
            ModelResponse(
                text=(
                    '{"route":"chat","confidence":1.0,'
                    '"task_input":null,"revision_instructions":null}'
                ),
                tool_calls=(),
                continuation_token="coordinator-response",
            ),
            ModelResponse(
                text="Exact graph-backed response",
                tool_calls=(),
                continuation_token="talker-response",
            ),
        ]
    )
    service, store = build_conversation_service(model)

    messages = service.process_message("conversation-1", "Hello")

    assert len(messages) == 1
    assert messages[0].content.endswith("Exact graph-backed response")
    completed = store.load("conversation-1")
    assert completed is not None
    assert completed.recent_messages[-1] == messages[0]
    checkpoint = service._graph_runner.get_state(
        {"configurable": {"thread_id": "editorial:v1:conversation-1"}}
    )
    assert checkpoint.values["completed_conversation"] == completed


def test_external_composition_reuses_one_service_model_and_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NamedFakeModel([])
    monkeypatch.setattr(composition, "create_gemini_client_from_env", lambda: model)

    application = build_external_api_application("placeholder-token")

    assert isinstance(application, ExternalApiApplication)
    assert application.adapter._service is application.service
    assert application.adapter._runtime_queue is application.runtime_queue
    assert application.service._workflow._writer._model is model
    assert application.service._workflow._critic._model is model
    assert application.service._workflow._editor._model is model
    assert application.runtime_queue.capacity == DEFAULT_RUNTIME_QUEUE_CAPACITY
    assert application.model_name == model.model
    assert application.runtime_queue.stats().worker_running is False


def test_missing_telegram_token_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(
        LiveConfigurationError,
        match=r"^Required Telegram configuration is missing$",
    ):
        build_live_application_from_env()


def test_model_configuration_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "123456:placeholder-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)

    def fail_model() -> object:
        raise ValueError("private provider diagnostics")

    monkeypatch.setattr(composition, "create_gemini_client_from_env", fail_model)

    with pytest.raises(LiveConfigurationError) as caught:
        build_live_application_from_env()

    assert str(caught.value) == "Required model configuration is missing or invalid"
    assert token not in str(caught.value)
    assert "private provider diagnostics" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_live_application_uses_real_adapter_and_sequential_telegram_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:placeholder-token")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "false")

    model = NamedFakeModel([])
    monkeypatch.setattr(composition, "create_gemini_client_from_env", lambda: model)

    live = build_live_application_from_env()

    assert isinstance(live.telegram, Application)
    assert isinstance(live.service, ConversationService)
    assert isinstance(live.store, InMemoryConversationStateStore)
    assert live.adapter._service is live.service
    assert isinstance(live.runtime_queue, RuntimeQueue)
    assert live.adapter._runtime_queue is live.runtime_queue
    assert live.runtime_queue.capacity == DEFAULT_RUNTIME_QUEUE_CAPACITY
    assert live.model_name == "safe-test-model"
    assert live.telegram.update_processor.max_concurrent_updates == 1
    assert len(live.telegram.handlers[0]) == 2
    assert live.telegram.post_init == live.adapter.start_runtime
    assert live.telegram.post_shutdown == live.adapter.close_runtime
    assert live.heartbeat is None
    assert live.adapter._heartbeat_store is None
    assert live.adapter._heartbeat_scheduler is None


def test_invalid_telegram_configuration_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "private-invalid-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(
        composition,
        "create_gemini_client_from_env",
        lambda: NamedFakeModel([]),
    )
    monkeypatch.setattr(
        composition,
        "build_telegram_application",
        lambda **_: (_ for _ in ()).throw(ValueError(f"invalid {token}")),
    )

    with pytest.raises(
        LiveConfigurationError,
        match=r"^Telegram configuration is invalid$",
    ) as caught:
        build_live_application_from_env()

    assert token not in str(caught.value)
    assert caught.value.__cause__ is None


def test_separate_compositions_do_not_share_in_memory_state() -> None:
    first_service, first_store = build_conversation_service(NamedFakeModel([]))
    second_service, second_store = build_conversation_service(NamedFakeModel([]))

    assert first_service is not second_service
    assert first_store is not second_store
    assert first_store.load("conversation-1") is None
    assert second_store.load("conversation-1") is None


def test_enabled_heartbeat_reuses_shared_model_queue_and_bot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    database = tmp_path / "heartbeat.db"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:placeholder-token")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "true")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_DB_PATH", str(database))
    monkeypatch.setenv("EDITORIAL_ADMIN_TELEGRAM_CHAT_ID", "-100123")
    model = NamedFakeModel([])
    monkeypatch.setattr(composition, "create_gemini_client_from_env", lambda: model)

    live = build_live_application_from_env()

    assert live.heartbeat is not None
    heartbeat = live.heartbeat
    assert isinstance(heartbeat.store, SQLiteHeartbeatResultStore)
    assert isinstance(heartbeat.evaluation_service, HeartbeatEvaluationService)
    assert isinstance(heartbeat.collector, OperationalSnapshotCollector)
    assert isinstance(heartbeat.runner, HeartbeatRunner)
    assert isinstance(heartbeat.scheduler, HeartbeatScheduler)
    assert heartbeat.admin_agent._model is model
    assert heartbeat.runner._runtime_queue is live.runtime_queue
    assert heartbeat.collector._runtime_queue is live.runtime_queue
    assert heartbeat.notifier._bot is live.telegram.bot
    assert live.adapter._heartbeat_store is heartbeat.store
    assert live.adapter._heartbeat_scheduler is heartbeat.scheduler
    assert not database.exists()


def test_combined_composition_reuses_exact_live_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:placeholder-token")
    monkeypatch.setenv("EDITORIAL_EXTERNAL_API_TOKEN", "placeholder-api-token")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "true")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_DB_PATH", str(tmp_path / "heartbeat.db"))
    monkeypatch.setenv("EDITORIAL_ADMIN_TELEGRAM_CHAT_ID", "-100123")
    model = NamedFakeModel([])
    model_constructions = 0

    def build_model() -> NamedFakeModel:
        nonlocal model_constructions
        model_constructions += 1
        return model

    monkeypatch.setattr(composition, "create_gemini_client_from_env", build_model)

    combined = build_combined_live_application_from_env()

    assert isinstance(combined, CombinedLiveApplication)
    assert model_constructions == 1
    assert combined.external_adapter._service is combined.live.service
    assert combined.external_adapter._runtime_queue is combined.live.runtime_queue
    assert combined.live.adapter._runtime_queue is combined.live.runtime_queue
    assert combined.live.heartbeat is not None
    assert combined.live.heartbeat.collector._runtime_queue is combined.live.runtime_queue
    assert combined.live.heartbeat.runner._runtime_queue is combined.live.runtime_queue
    assert combined.live.telegram.post_init == combined.lifecycle.start
    assert combined.live.telegram.post_shutdown == combined.lifecycle.close


def test_combined_external_jobs_update_shared_queue_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:placeholder-token")
    monkeypatch.setenv("EDITORIAL_EXTERNAL_API_TOKEN", "placeholder-api-token")
    monkeypatch.setenv("EDITORIAL_HEARTBEAT_ENABLED", "false")
    monkeypatch.setattr(
        composition,
        "create_gemini_client_from_env",
        lambda: NamedFakeModel([]),
    )
    combined = build_combined_live_application_from_env()

    async def scenario() -> tuple[object, object]:
        await combined.runtime_queue.start()
        monkeypatch.setattr(
            combined.live.service,
            "process_brief",
            lambda brief: SimpleNamespace(working_draft="Completed copy"),
        )
        success = await combined.external_adapter.handle(
            headers={
                "Authorization": "Bearer placeholder-api-token",
                "Content-Type": "application/json",
                "Idempotency-Key": "success",
            },
            body=json.dumps({"brief": "Safe brief"}).encode(),
        )

        def fail(brief: str) -> object:
            del brief
            raise RuntimeError("PRIVATE-MODEL-DIAGNOSTIC")

        monkeypatch.setattr(combined.live.service, "process_brief", fail)
        failure = await combined.external_adapter.handle(
            headers={
                "Authorization": "Bearer placeholder-api-token",
                "Content-Type": "application/json",
                "Idempotency-Key": "failure",
            },
            body=json.dumps({"brief": "Another safe brief"}).encode(),
        )
        stats = combined.runtime_queue.stats().for_source(RuntimeJobSource.EXTERNAL)
        await combined.runtime_queue.close()
        return success, failure, stats

    success, failure, stats = asyncio.run(scenario())

    assert success.status_code == 200
    assert failure.status_code == 500
    assert stats.completed_jobs == 1
    assert stats.failed_jobs == 1


def test_combined_lifecycle_starts_and_closes_shared_runtime_once() -> None:
    events: list[str] = []

    class Adapter:
        async def start_runtime(self, application: object) -> None:
            del application
            events.append("runtime-start")

        async def close_runtime(self, application: object) -> None:
            del application
            events.append("runtime-close")

    class ExternalAdapter:
        def stop_accepting(self) -> None:
            events.append("stop-accepting")

    class Server:
        def __init__(self, address, *, adapter, loop) -> None:
            del adapter, loop
            assert address == ("127.0.0.1", 8080)
            events.append("server-created")

        def serve_forever(self) -> None:
            events.append("server-run")

        def shutdown(self) -> None:
            events.append("server-shutdown")

        def server_close(self) -> None:
            events.append("server-close")

    lifecycle = CombinedRuntimeLifecycle(
        live=SimpleNamespace(adapter=Adapter()),  # type: ignore[arg-type]
        external_adapter=ExternalAdapter(),  # type: ignore[arg-type]
        configuration=SimpleNamespace(host="127.0.0.1", port=8080),  # type: ignore[arg-type]
        server_factory=Server,  # type: ignore[arg-type]
    )

    async def scenario() -> None:
        application = object()
        await lifecycle.start(application)  # type: ignore[arg-type]
        await lifecycle.close(application)  # type: ignore[arg-type]

    asyncio.run(scenario())

    assert events.count("runtime-start") == 1
    assert events.count("runtime-close") == 1
    assert set(events) == {
        "server-created",
        "runtime-start",
        "stop-accepting",
        "server-run",
        "server-shutdown",
        "server-close",
        "runtime-close",
    }
    assert events.index("server-created") < events.index("runtime-start")
    assert events.index("stop-accepting") < events.index("runtime-close")
    assert events.index("server-close") < events.index("runtime-close")


def test_combined_bind_failure_starts_no_runtime() -> None:
    events: list[str] = []

    class Adapter:
        async def start_runtime(self, application: object) -> None:
            del application
            events.append("runtime-start")

    lifecycle = CombinedRuntimeLifecycle(
        live=SimpleNamespace(adapter=Adapter()),  # type: ignore[arg-type]
        external_adapter=object(),  # type: ignore[arg-type]
        configuration=SimpleNamespace(host="127.0.0.1", port=8080),  # type: ignore[arg-type]
        server_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("PRIVATE-BIND-DIAGNOSTIC")
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        LiveConfigurationError,
        match=r"^External HTTP server configuration is invalid$",
    ) as caught:
        asyncio.run(lifecycle.start(object()))  # type: ignore[arg-type]

    assert caught.value.__cause__ is None
    assert events == []


@pytest.mark.parametrize("token", [None, "", "   "])
def test_combined_composition_requires_external_token(
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
) -> None:
    if token is None:
        monkeypatch.delenv("EDITORIAL_EXTERNAL_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EDITORIAL_EXTERNAL_API_TOKEN", token)

    with pytest.raises(
        LiveConfigurationError,
        match=r"^Required external API configuration is missing$",
    ):
        build_combined_live_application_from_env()
