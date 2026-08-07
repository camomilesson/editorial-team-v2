#!/usr/bin/env python3
"""Seed deterministic development artifacts through the production repository."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from editorial_team.app.artifact_config import load_artifact_configuration
from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    ParagraphChunker,
    SQLiteArtifactStore,
    artifact_id_for,
    content_sha256,
)
from editorial_team.contracts.common import parse_utc_timestamp

DEFAULT_FIXTURE_PATH = Path("evaluation/fixtures/sample_artifacts.json")


def load_fixture(path: Path) -> tuple[EditorialArtifact, ...]:
    """Load and validate deterministic fixture records as production models."""

    try:
        values: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact fixture could not be loaded") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("Artifact fixture must be a non-empty array")
    artifacts: list[EditorialArtifact] = []
    required = {
        "task_id",
        "producer",
        "created_at",
        "conversation_id",
        "user_request",
        "content",
    }
    for value in values:
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Artifact fixture record has invalid fields")
        try:
            producer = ArtifactProducer(value["producer"])
            content = value["content"]
            task_id = value["task_id"]
            artifacts.append(
                EditorialArtifact(
                    artifact_id=artifact_id_for(task_id, producer),
                    task_id=task_id,
                    producer=producer,
                    created_at=parse_utc_timestamp(value["created_at"], "created_at"),
                    conversation_id=value["conversation_id"],
                    user_request=value["user_request"],
                    content=content,
                    content_sha256=content_sha256(content),
                )
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("Artifact fixture record is invalid") from None
    return tuple(artifacts)


def seed(database_path: Path, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> int:
    """Persist fixture runs through the real store and chunker."""

    grouped: dict[str, list[EditorialArtifact]] = defaultdict(list)
    for artifact in load_fixture(fixture_path):
        grouped[artifact.task_id].append(artifact)
    store = SQLiteArtifactStore(database_path, chunker=ParagraphChunker())
    store.initialize()
    try:
        for task_id in sorted(grouped):
            run = sorted(
                grouped[task_id],
                key=lambda item: 0 if item.producer is ArtifactProducer.WRITER else 1,
            )
            store.save_run(tuple(run))
    finally:
        store.close()
    return sum(len(run) for run in grouped.values())


def main() -> None:
    """Seed the configured artifact database from a JSON fixture."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    arguments = parser.parse_args()
    database = arguments.database or load_artifact_configuration().database_path
    count = seed(database, arguments.fixture)
    print(f"Seeded {count} editorial artifacts.")


if __name__ == "__main__":
    main()
