"""Configuration for the dedicated editorial artifact database."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACT_DB_PATH = Path("runtime_data/editorial_artifacts.db")


class ArtifactConfigurationError(RuntimeError):
    """Editorial artifact configuration is invalid."""


@dataclass(frozen=True)
class ArtifactConfiguration:
    """Validated non-secret artifact persistence settings."""

    database_path: Path = DEFAULT_ARTIFACT_DB_PATH


def load_artifact_configuration() -> ArtifactConfiguration:
    """Load the local artifact path without exposing invalid raw values."""

    raw_path = os.getenv("EDITORIAL_ARTIFACT_DB_PATH", str(DEFAULT_ARTIFACT_DB_PATH))
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ArtifactConfigurationError("Artifact database path is invalid")
    return ArtifactConfiguration(Path(raw_path.strip()))
