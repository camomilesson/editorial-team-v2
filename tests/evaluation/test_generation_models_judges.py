import json

import pytest

from editorial_team.evaluation.generation_judges import (
    GenerationMetric,
    JudgeError,
    StructuredGenerationJudge,
    judge_prompt,
)
from editorial_team.evaluation.generation_models import (
    GenerationContext,
    GenerationError,
    GroundedGenerator,
    generation_prompt,
)
from editorial_team.models import FakeModelClient, ModelResponse

CONTEXTS = (
    GenerationContext("chunk-2", "artifact-2", "second untrusted text"),
    GenerationContext("chunk-1", "artifact-1", "first untrusted text"),
)


def response(text: str) -> ModelResponse:
    return ModelResponse(text, (), None)


def test_generator_preserves_exact_order_and_marks_context_untrusted() -> None:
    prompt = generation_prompt("question", CONTEXTS)
    assert prompt.index("chunk-2") < prompt.index("chunk-1")
    assert "UNTRUSTED APPLICATION DATA" in prompt
    assert "corpus does not provide the answer" in prompt
    assert "rerank" not in prompt.lower()
    model = FakeModelClient([response("grounded answer")])
    assert GroundedGenerator(model).generate("question", CONTEXTS) == "grounded answer"


def test_generator_sanitizes_malformed_output() -> None:
    with pytest.raises(GenerationError, match="invalid output"):
        GroundedGenerator(FakeModelClient([response(" ")])).generate("question", ())


@pytest.mark.parametrize("metric", list(GenerationMetric))
def test_judges_use_separate_metric_rubrics_and_hide_condition(metric: GenerationMetric) -> None:
    prompt = judge_prompt(
        metric,
        query="q",
        candidate_answer="candidate",
        golden_answer="gold",
        retrieved_contexts=CONTEXTS,
        golden_contexts=CONTEXTS[:1],
    )
    assert metric.value in prompt
    assert "rerank" not in prompt.lower()
    model = FakeModelClient([response(json.dumps({"score": 0.75, "reason": "bounded"}))])
    score = StructuredGenerationJudge(model).judge(
        metric,
        query="q",
        candidate_answer="candidate",
        golden_answer="gold",
        retrieved_contexts=CONTEXTS,
        golden_contexts=CONTEXTS[:1],
    )
    assert score.score == 0.75
    assert model.requests[0].structured_output is not None


@pytest.mark.parametrize(
    "payload",
    [
        "bad",
        '{"score": 2, "reason": "bad"}',
        '{"score": 1}',
        '{"score": 1, "reason": "x", "extra": 1}',
    ],
)
def test_judge_rejects_malformed_scores_fields_and_json(payload: str) -> None:
    with pytest.raises(JudgeError, match="invalid structured output"):
        StructuredGenerationJudge(FakeModelClient([response(payload)])).judge(
            GenerationMetric.FAITHFULNESS,
            query="q",
            candidate_answer="a",
            golden_answer="g",
            retrieved_contexts=(),
            golden_contexts=(),
        )


def test_judge_provider_failure_is_sanitized() -> None:
    with pytest.raises(JudgeError, match="model call failed"):
        StructuredGenerationJudge(FakeModelClient([])).judge(
            GenerationMetric.CONTEXT_RECALL,
            query="q",
            candidate_answer="a",
            golden_answer="g",
            retrieved_contexts=(),
            golden_contexts=(),
        )
