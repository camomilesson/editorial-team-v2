import hashlib
from datetime import UTC, datetime
from pathlib import Path

from editorial_team.app.retrieval_config import RetrievalConfiguration
from editorial_team.evaluation.generation_cache import JudgeCache
from editorial_team.evaluation.generation_cases import load_generation_cases
from editorial_team.evaluation.generation_judges import JudgeScore
from editorial_team.evaluation.generation_reporting import render_generation_report
from editorial_team.evaluation.generation_runner import run_generation_evaluation
from editorial_team.evaluation.retrieval_cases import load_corpus

ROOT = Path(__file__).resolve().parents[2]


class Embeddings:
    model_id = "fake-embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._value(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._value(text)

    @staticmethod
    def _value(text: str) -> list[float]:
        return [float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:6]]


class Reranker:
    model_id = "fake-reranker"

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [float(index) for index, _ in enumerate(passages)]


class Generator:
    calls = 0

    def generate(self, query: str, contexts: tuple) -> str:
        self.calls += 1
        return f"Answer to {query} from {len(contexts)} contexts"


class Judge:
    calls = 0

    def judge(self, metric: object, **kwargs: object) -> JudgeScore:
        del metric, kwargs
        self.calls += 1
        return JudgeScore(0.5, "fake fixed rubric")


def test_runner_executes_full_on_subset_off_aggregates_and_cache(tmp_path: Path) -> None:
    cases_path = ROOT / "evaluation/generation/cases.jsonl"
    corpus_path = ROOT / "evaluation/retrieval/corpus.json"
    cases = load_generation_cases(cases_path)
    generator, judge = Generator(), Judge()
    result = run_generation_evaluation(
        database_path=tmp_path / "eval.db",
        corpus=load_corpus(corpus_path),
        cases=cases,
        corpus_path=corpus_path,
        cases_path=cases_path,
        embeddings=Embeddings(),
        reranker=Reranker(),
        generator=generator,  # type: ignore[arg-type]
        judge=judge,  # type: ignore[arg-type]
        cache=JudgeCache(tmp_path / "cache.json"),
        generator_model="fake-generator",
        judge_model="fake-judge",
        configuration=RetrievalConfiguration(),
        run_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    assert len(result["conditions"]["on"]["cases"]) == 20
    assert len(result["conditions"]["off"]["cases"]) >= 8
    assert generator.calls == 20 + len(result["conditions"]["off"]["cases"])
    assert result["cache"]["hits"] + result["cache"]["misses"] == generator.calls * 4
    assert judge.calls == result["cache"]["misses"]
    assert set(result["conditions"]["on"]["overall"].values()) == {0.5}
    assert result["corpus_sha256"] and result["case_set_sha256"]
    report = render_generation_report(result)
    assert "| rerank on | 0.5000 | 0.5000 | 0.5000 | 0.5000 |" in report
    assert "Judge bias and limitations" in report


def test_application_composition_remains_evaluation_free() -> None:
    source = (ROOT / "src/editorial_team/app/composition.py").read_text()
    assert "editorial_team.evaluation" not in source
