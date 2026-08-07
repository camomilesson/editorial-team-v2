from pathlib import Path

from editorial_team.evaluation.generation_cases import (
    OUT_OF_CORPUS,
    load_generation_cases,
    resolve_generation_anchors,
)
from editorial_team.evaluation.retrieval_cases import load_corpus

ROOT = Path(__file__).resolve().parents[2]


def test_generation_cases_have_scale_categories_subset_and_stable_anchors() -> None:
    cases = load_generation_cases(ROOT / "evaluation/generation/cases.jsonl")
    corpus = load_corpus(ROOT / "evaluation/retrieval/corpus.json")
    first = resolve_generation_anchors(corpus, cases)
    assert len(cases) == 20
    assert len({case.failure_category for case in cases}) >= 5
    assert sum(case.failure_category == OUT_OF_CORPUS for case in cases) == 3
    assert sum(case.compare_rerank_off for case in cases) >= 8
    assert first == resolve_generation_anchors(corpus, cases)
    assert all(first[case.case_id] for case in cases if not case.expected_out_of_corpus)
    assert all(not first[case.case_id] for case in cases if case.expected_out_of_corpus)
