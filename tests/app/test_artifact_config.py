"""Artifact database configuration tests."""

from pathlib import Path

import pytest

from editorial_team.app.artifact_config import (
    DEFAULT_ARTIFACT_DB_PATH,
    ArtifactConfigurationError,
    load_artifact_configuration,
)


def test_artifact_path_defaults_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITORIAL_ARTIFACT_DB_PATH", raising=False)
    assert load_artifact_configuration().database_path == DEFAULT_ARTIFACT_DB_PATH
    monkeypatch.setenv("EDITORIAL_ARTIFACT_DB_PATH", "custom/artifacts.db")
    assert load_artifact_configuration().database_path == Path("custom/artifacts.db")


def test_blank_artifact_path_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_ARTIFACT_DB_PATH", " ")
    with pytest.raises(ArtifactConfigurationError, match="path is invalid"):
        load_artifact_configuration()
