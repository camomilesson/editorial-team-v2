"""Validated fixed generation-evaluation cases anchored to the retrieval corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from editorial_team.artifacts import ParagraphChunker
from editorial_team.evaluation.retrieval_cases import CorpusArtifact, GoldenAnchor

OUT_OF_CORPUS = "out_of_corpus_required_abstention"


@dataclass(frozen=True)
class GenerationCase:
    case_id: str
    query: str
    golden_answer: str
    golden_context_anchors: tuple[GoldenAnchor, ...]
    failure_category: str
    tags: tuple[str, ...]
    expected_out_of_corpus: bool
    compare_rerank_off: bool


def load_generation_cases(path: Path) -> tuple[GenerationCase, ...]:
    cases: list[GenerationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value: Any = json.loads(line)
            case = GenerationCase(
                case_id=value["case_id"],
                query=value["query"],
                golden_answer=value["golden_answer"],
                golden_context_anchors=tuple(
                    GoldenAnchor(**anchor) for anchor in value["golden_context_anchors"]
                ),
                failure_category=value["failure_category"],
                tags=tuple(value["tags"]),
                expected_out_of_corpus=value["expected_out_of_corpus"],
                compare_rerank_off=value["compare_rerank_off"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("generation case is invalid") from None
        if not case.case_id.strip() or not case.query.strip() or not case.golden_answer.strip():
            raise ValueError("generation case text is invalid")
        if case.expected_out_of_corpus != (case.failure_category == OUT_OF_CORPUS):
            raise ValueError("out-of-corpus declaration is inconsistent")
        if case.expected_out_of_corpus == bool(case.golden_context_anchors):
            raise ValueError("generation golden contexts are inconsistent")
        cases.append(case)
    if len(cases) < 20 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("at least 20 unique generation cases are required")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def resolve_generation_anchors(
    corpus: tuple[CorpusArtifact, ...], cases: tuple[GenerationCase, ...]
) -> dict[str, tuple[str, ...]]:
    artifacts = {item.fixture_id: item.artifact for item in corpus}
    chunker = ParagraphChunker()
    output: dict[str, tuple[str, ...]] = {}
    for case in cases:
        resolved: list[str] = []
        for anchor in case.golden_context_anchors:
            artifact = artifacts.get(anchor.artifact_fixture_id)
            if artifact is None:
                raise ValueError("generation anchor artifact is missing")
            chunks = chunker.chunk(artifact)
            if not 0 <= anchor.chunk_ordinal < len(chunks):
                raise ValueError("generation anchor chunk is missing")
            chunk = chunks[anchor.chunk_ordinal]
            if anchor.required_text not in chunk.content:
                raise ValueError("generation anchor text is missing")
            resolved.append(chunk.chunk_id)
        output[case.case_id] = tuple(resolved)
    return output
