from __future__ import annotations

import pytest
from telegram.ext import Application

from editorial_team.app import composition
from editorial_team.app.composition import (
    RECENT_MESSAGE_LIMIT,
    LiveConfigurationError,
    build_conversation_service,
    build_live_application_from_env,
)
from editorial_team.conversation import ConversationService, InMemoryConversationStateStore
from editorial_team.models import FakeModelClient


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
    model = NamedFakeModel([])
    monkeypatch.setattr(composition, "create_gemini_client_from_env", lambda: model)

    live = build_live_application_from_env()

    assert isinstance(live.telegram, Application)
    assert isinstance(live.service, ConversationService)
    assert isinstance(live.store, InMemoryConversationStateStore)
    assert live.adapter._service is live.service
    assert live.model_name == "safe-test-model"
    assert live.telegram.update_processor.max_concurrent_updates == 1
    assert len(live.telegram.handlers[0]) == 2


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
