"""Focused production-composition ownership tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from editorial_team.app import composition
from editorial_team.app.checkpoint_config import CheckpointConfiguration
from editorial_team.app.heartbeat_config import HeartbeatConfiguration
from editorial_team.app.telegram_config import TelegramConfiguration
from editorial_team.domain.conversation import Message, MessageRole


class Model:
    model = "fake-model"

    def respond(self, request: object) -> object:
        del request
        raise AssertionError("model must not be called during composition")


class Runner:
    def __init__(self) -> None:
        self.invocations = 0

    def invoke(self, state: object, config: object) -> dict[str, object]:
        del state, config
        self.invocations += 1
        return {
            "assistant_messages": (
                Message(
                    "message-1",
                    "telegram-chat-1",
                    MessageRole.ASSISTANT,
                    "Talker\n\nHello",
                    composition.datetime.now(composition.UTC),
                ),
            )
        }


class Builder:
    def __init__(self, runner: Runner, counts: dict[str, int]) -> None:
        self.runner = runner
        self.counts = counts

    def compile(self, *, checkpointer: object) -> Runner:
        del checkpointer
        self.counts["compile"] += 1
        return self.runner


class ArtifactStore:
    def __init__(self, path: Path, *, chunker: object, events: list[str]) -> None:
        del chunker
        self.path = path
        self.events = events
        events.append("constructed")

    def initialize(self) -> None:
        self.events.append("initialized")

    def close(self) -> None:
        self.events.append("closed")


def test_composition_builds_checkpointer_and_graph_once_for_repeated_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    counts = {"checkpointer": 0, "compile": 0, "close": 0}
    runner = Runner()

    def checkpointer(path: Path, *, busy_timeout_seconds: float) -> tuple[object, object]:
        assert path == tmp_path / "state.db"
        assert busy_timeout_seconds == 0.25
        counts["checkpointer"] += 1
        return object(), lambda: counts.__setitem__("close", counts["close"] + 1)

    monkeypatch.setattr(composition, "create_sqlite_checkpointer", checkpointer)
    monkeypatch.setattr(
        composition,
        "build_parent_graph",
        lambda **kwargs: Builder(runner, counts),
    )

    service = composition.build_conversation_service(
        Model(), tmp_path / "state.db", busy_timeout_seconds=0.25
    )
    service.process_message("telegram-chat-1", "one")
    service.process_message("telegram-chat-1", "two")
    service.close()
    service.close()

    assert counts == {"checkpointer": 1, "compile": 1, "close": 1}
    assert runner.invocations == 2


def test_graph_construction_failure_closes_checkpointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[bool] = []
    artifact_events: list[str] = []
    monkeypatch.setattr(
        composition,
        "SQLiteArtifactStore",
        lambda path, *, chunker: ArtifactStore(
            path, chunker=chunker, events=artifact_events
        ),
    )
    monkeypatch.setattr(
        composition,
        "create_sqlite_checkpointer",
        lambda path, **kwargs: (object(), lambda: closed.append(True)),
    )
    monkeypatch.setattr(
        composition,
        "build_parent_graph",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        composition.build_conversation_service(Model(), tmp_path / "state.db")

    assert closed == [True]
    assert artifact_events == ["constructed", "initialized", "closed"]


def test_artifact_store_is_owned_once_and_closed_with_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    runner = Runner()
    monkeypatch.setattr(
        composition,
        "SQLiteArtifactStore",
        lambda path, *, chunker: ArtifactStore(path, chunker=chunker, events=events),
    )
    monkeypatch.setattr(
        composition,
        "create_sqlite_checkpointer",
        lambda path, **kwargs: (object(), lambda: events.append("checkpoint-closed")),
    )
    monkeypatch.setattr(
        composition,
        "build_parent_graph",
        lambda **kwargs: Builder(runner, {"compile": 0}),
    )
    service = composition.build_conversation_service(
        Model(),
        tmp_path / "state.db",
        artifact_path=tmp_path / "artifacts.db",
    )
    service.close()
    service.close()
    assert events == ["constructed", "initialized", "checkpoint-closed", "closed"]


def test_heartbeat_construction_failure_closes_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[bool] = []
    service = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder")
    monkeypatch.setattr(
        composition,
        "load_telegram_configuration",
        lambda: TelegramConfiguration(frozenset({123})),
    )
    monkeypatch.setattr(
        composition,
        "load_checkpoint_configuration",
        lambda: CheckpointConfiguration(tmp_path / "state.db", 0.1),
    )
    monkeypatch.setattr(
        composition,
        "load_heartbeat_configuration",
        lambda: HeartbeatConfiguration(enabled=True, maintainer_chat_id=123),
    )
    monkeypatch.setattr(composition, "create_gemini_client_from_env", Model)
    monkeypatch.setattr(composition, "create_gemini_chat_model_from_env", object)
    monkeypatch.setattr(composition, "build_conversation_service", lambda *a, **k: service)
    monkeypatch.setattr(
        composition,
        "build_telegram_application",
        lambda **kwargs: SimpleNamespace(bot=object()),
    )
    monkeypatch.setattr(
        composition,
        "SQLiteHeartbeatResultStore",
        lambda path: (_ for _ in ()).throw(RuntimeError("heartbeat failed")),
    )

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        composition.build_live_application_from_env()

    assert closed == [True]


def test_adapter_construction_failure_closes_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[bool] = []
    service = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "placeholder")
    monkeypatch.setattr(
        composition,
        "load_telegram_configuration",
        lambda: TelegramConfiguration(frozenset({123})),
    )
    monkeypatch.setattr(
        composition,
        "load_checkpoint_configuration",
        lambda: CheckpointConfiguration(tmp_path / "state.db", 0.1),
    )
    monkeypatch.setattr(
        composition,
        "load_heartbeat_configuration",
        lambda: HeartbeatConfiguration(enabled=False),
    )
    monkeypatch.setattr(composition, "create_gemini_client_from_env", Model)
    monkeypatch.setattr(composition, "create_gemini_chat_model_from_env", object)
    monkeypatch.setattr(composition, "build_conversation_service", lambda *a, **k: service)
    monkeypatch.setattr(
        composition,
        "TelegramAdapter",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("adapter failed")),
    )

    with pytest.raises(composition.LiveConfigurationError, match="Telegram configuration"):
        composition.build_live_application_from_env()

    assert closed == [True]
