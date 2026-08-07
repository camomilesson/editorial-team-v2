from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from editorial_team.app.retrieval_config import RetrievalConfiguration
from editorial_team.artifacts import ParagraphChunker
from editorial_team.evaluation.retrieval_cases import load_cases, load_corpus
from editorial_team.evaluation.retrieval_reporting import render_markdown
from editorial_team.evaluation.retrieval_runner import run_evaluation

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "evaluation/retrieval/corpus.json"
CASES = ROOT / "evaluation/retrieval/cases.jsonl"


class FakeEmbeddings:
    model_id = "deterministic-hash-embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        return [float(value + 1) for value in digest[:8]]


class FakeReranker:
    model_id = "deterministic-token-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        return [
            float(len(query_tokens.intersection(passage.lower().split())))
            for passage in passages
        ]


def evaluated(tmp_path: Path) -> dict:
    return run_evaluation(
        database_path=tmp_path / "eval.db",
        corpus=load_corpus(CORPUS),
        cases=load_cases(CASES),
        corpus_path=CORPUS,
        cases_path=CASES,
        embeddings=FakeEmbeddings(),
        reranker=FakeReranker(),
        configuration=RetrievalConfiguration(),
        run_at=datetime(2026, 4, 1, tzinfo=UTC),
    )


def test_runner_uses_identical_corpus_cases_and_exact_final_order_for_both_modes(
    tmp_path: Path,
) -> None:
    result = evaluated(tmp_path)
    assert result["configuration"]["rerank_modes"] == ["off", "on"]
    assert result["case_set"]["case_count"] == 12
    off = result["conditions"]["off"]["cases"]
    on = result["conditions"]["on"]["cases"]
    assert [row["case_id"] for row in off] == [row["case_id"] for row in on]
    for condition in result["conditions"].values():
        for case in condition["cases"]:
            assert case["final_rankings"] == case["stage_rankings"]["final"]
            assert len(case["final_rankings"]) == len(set(case["final_rankings"]))


def test_cross_conversation_distractors_never_appear_and_empty_case_is_na(
    tmp_path: Path,
) -> None:
    result = evaluated(tmp_path)
    chunker = ParagraphChunker()
    distractors = {
        chunk.chunk_id
        for item in load_corpus(CORPUS)
        if item.artifact.conversation_id == "eval-retrieval-other"
        for chunk in chunker.chunk(item.artifact)
    }
    for condition in result["conditions"].values():
        assert condition["empty_golden_count"] == 1
        for case in condition["cases"]:
            assert case["metrics"] is None if case["empty_golden"] else case["metrics"] is not None
            assert not distractors.intersection(case["final_rankings"])


def test_report_tables_match_computed_aggregates(tmp_path: Path) -> None:
    result = evaluated(tmp_path)
    report = render_markdown(result)
    row = result["conditions"]["off"]["aggregate_metrics"]["1"]
    assert (
        f"| off | 1 | {row['hit_rate']:.4f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['mrr']:.4f} | {row['ndcg']:.4f} |"
    ) in report
    assert "Empty-golden cases: 1" in report


def test_application_composition_does_not_import_evaluation_package() -> None:
    source = (ROOT / "src/editorial_team/app/composition.py").read_text(encoding="utf-8")
    assert "editorial_team.evaluation" not in source
