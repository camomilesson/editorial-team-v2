"""Fixed-corpus and golden-anchor loading for retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    ParagraphChunker,
    artifact_id_for,
    content_sha256,
)
from editorial_team.contracts.common import parse_utc_timestamp


@dataclass(frozen=True)
class CorpusArtifact:
    fixture_id: str
    artifact: EditorialArtifact


@dataclass(frozen=True)
class GoldenAnchor:
    artifact_fixture_id: str
    chunk_ordinal: int
    required_text: str


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    description: str
    query: str
    conversation_id: str
    created_from: datetime | None
    created_to: datetime | None
    prefer_recent: bool
    golden_anchors: tuple[GoldenAnchor, ...]
    tags: tuple[str, ...]
    empty_golden: bool


def load_corpus(path: Path) -> tuple[CorpusArtifact, ...]:
    values = _json(path)
    if not isinstance(values, list) or not values:
        raise ValueError("corpus must be a non-empty array")
    output: list[CorpusArtifact] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("corpus record must be an object")
        fixture_id = value.get("fixture_id")
        task_id = value.get("task_id")
        content = value.get("content")
        try:
            producer = ArtifactProducer(value.get("producer"))
            artifact = EditorialArtifact(
                artifact_id=artifact_id_for(task_id, producer),
                task_id=task_id,
                producer=producer,
                created_at=parse_utc_timestamp(value.get("created_at"), "created_at"),
                conversation_id=value.get("conversation_id"),
                user_request=value.get("user_request"),
                content=content,
                content_sha256=content_sha256(content),
            )
            if not isinstance(fixture_id, str) or not fixture_id.strip():
                raise ValueError("fixture_id is invalid")
        except (TypeError, ValueError):
            raise ValueError("corpus record is invalid") from None
        output.append(CorpusArtifact(fixture_id.strip(), artifact))
    if len({item.fixture_id for item in output}) != len(output):
        raise ValueError("fixture artifact IDs must be unique")
    run_members = {(item.artifact.task_id, item.artifact.producer) for item in output}
    if len(run_members) != len(output):
        raise ValueError("fixture task and producer pairs must be unique")
    return tuple(output)


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    rows = path.read_text(encoding="utf-8").splitlines()
    cases: list[RetrievalCase] = []
    for row in rows:
        if not row.strip():
            continue
        try:
            value: Any = json.loads(row)
            anchors = tuple(GoldenAnchor(**anchor) for anchor in value["golden_anchors"])
            case = RetrievalCase(
                case_id=value["case_id"],
                description=value["description"],
                query=value["query"],
                conversation_id=value["conversation_id"],
                created_from=_optional_time(value["created_from"]),
                created_to=_optional_time(value["created_to"]),
                prefer_recent=value["prefer_recent"],
                golden_anchors=anchors,
                tags=tuple(value["tags"]),
                empty_golden=value["empty_golden"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("retrieval case is invalid") from None
        if case.empty_golden == bool(case.golden_anchors):
            raise ValueError("empty-golden declaration is inconsistent")
        cases.append(case)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be present and unique")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def resolve_golden_chunks(
    corpus: tuple[CorpusArtifact, ...],
    cases: tuple[RetrievalCase, ...],
    chunker: ParagraphChunker,
) -> dict[str, tuple[str, ...]]:
    """Resolve and verify stable content anchors through the production chunker."""

    artifacts = {item.fixture_id: item.artifact for item in corpus}
    resolved: dict[str, tuple[str, ...]] = {}
    for case in cases:
        ids: list[str] = []
        for anchor in case.golden_anchors:
            artifact = artifacts.get(anchor.artifact_fixture_id)
            if artifact is None:
                raise ValueError("golden anchor artifact is missing")
            chunks = chunker.chunk(artifact)
            if not 0 <= anchor.chunk_ordinal < len(chunks):
                raise ValueError("golden anchor chunk is missing")
            chunk = chunks[anchor.chunk_ordinal]
            if anchor.required_text not in chunk.content:
                raise ValueError("golden anchor text is missing")
            ids.append(chunk.chunk_id)
        resolved[case.case_id] = tuple(ids)
    return resolved


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("JSON fixture could not be loaded") from exc


def _optional_time(value: Any) -> datetime | None:
    return None if value is None else parse_utc_timestamp(value, "case timestamp")
