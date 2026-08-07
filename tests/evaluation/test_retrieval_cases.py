from __future__ import annotations

import json
from pathlib import Path

import pytest

from editorial_team.artifacts import ParagraphChunker
from editorial_team.evaluation.retrieval_cases import (
    load_cases,
    load_corpus,
    resolve_golden_chunks,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "evaluation/retrieval/corpus.json"
CASES = ROOT / "evaluation/retrieval/cases.jsonl"


def test_fixed_corpus_and_cases_have_required_scale_variety_and_unique_ids() -> None:
    corpus = load_corpus(CORPUS)
    cases = load_cases(CASES)
    assert 25 <= len(corpus) <= 40
    assert len(cases) >= 10
    assert len({item.fixture_id for item in corpus}) == len(corpus)
    assert len({(item.artifact.task_id, item.artifact.producer) for item in corpus}) == len(corpus)
    assert sum(item.artifact.producer.value == "editor" for item in corpus) >= 2
    assert len({case.case_id for case in cases}) == len(cases)
    assert {case.conversation_id for case in cases} == {"eval-retrieval-main"}
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "exact_rare_term",
        "acronym",
        "semantic_paraphrase",
        "lexical_semantic_mismatch",
        "near_duplicate",
        "entity_ambiguity",
        "heading_specific",
        "date_lower_bound",
        "bounded_date_range",
        "recency_preference",
        "multi_relevant",
        "out_of_corpus",
    } <= tags


def test_all_golden_anchors_resolve_stably_and_contain_required_text() -> None:
    corpus = load_corpus(CORPUS)
    cases = load_cases(CASES)
    first = resolve_golden_chunks(corpus, cases, ParagraphChunker())
    second = resolve_golden_chunks(corpus, cases, ParagraphChunker())
    assert first == second
    assert all(first[case.case_id] for case in cases if not case.empty_golden)
    assert all(not first[case.case_id] for case in cases if case.empty_golden)


def test_long_fixture_produces_multiple_production_chunks() -> None:
    corpus = {item.fixture_id: item.artifact for item in load_corpus(CORPUS)}
    assert len(ParagraphChunker().chunk(corpus["long-market-report"])) >= 2


def test_empty_golden_case_with_anchor_is_rejected(tmp_path: Path) -> None:
    value = {
        "case_id": "bad",
        "description": "bad",
        "query": "bad",
        "conversation_id": "eval-retrieval-main",
        "created_from": None,
        "created_to": None,
        "prefer_recent": False,
        "golden_anchors": [
            {"artifact_fixture_id": "x", "chunk_ordinal": 0, "required_text": "x"}
        ],
        "tags": [],
        "empty_golden": True,
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_cases(path)
