#!/usr/bin/env python3
"""Run standalone judged generation evaluation with real local retrieval and Gemini."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from editorial_team.app.retrieval_config import load_retrieval_configuration
from editorial_team.artifacts.embeddings import SentenceTransformerEmbeddingModel
from editorial_team.artifacts.reranking import CrossEncoderReranker
from editorial_team.evaluation.generation_cache import JudgeCache
from editorial_team.evaluation.generation_cases import load_generation_cases
from editorial_team.evaluation.generation_judges import StructuredGenerationJudge
from editorial_team.evaluation.generation_models import GroundedGenerator
from editorial_team.evaluation.generation_reporting import write_generation_results
from editorial_team.evaluation.generation_runner import run_generation_evaluation
from editorial_team.evaluation.retrieval_cases import load_corpus
from editorial_team.gemini import DEFAULT_GEMINI_MODEL, GeminiModelClient

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "evaluation/retrieval/corpus.json"
CASES = Path(__file__).with_name("cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluation/outputs/generation_results.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "evaluation/outputs/generation_report.md"
    )
    arguments = parser.parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    fallback = os.getenv("AGENT_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
    generator_model = os.getenv("EDITORIAL_EVAL_GENERATOR_MODEL", "").strip() or fallback
    judge_model = os.getenv("EDITORIAL_EVAL_JUDGE_MODEL", "").strip() or fallback
    cache_path = Path(
        os.getenv("EDITORIAL_EVAL_CACHE_PATH", "evaluation/.cache/generation_judges.json")
    )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    database = arguments.database
    if database is None:
        temporary = tempfile.TemporaryDirectory(prefix="editorial-generation-eval-")
        database = Path(temporary.name) / "evaluation.db"
    configuration = load_retrieval_configuration()
    try:
        result = run_generation_evaluation(
            database_path=database,
            corpus=load_corpus(CORPUS),
            cases=load_generation_cases(CASES),
            corpus_path=CORPUS,
            cases_path=CASES,
            embeddings=SentenceTransformerEmbeddingModel(configuration.embedding_model),
            reranker=CrossEncoderReranker(configuration.reranker_model),
            generator=GroundedGenerator(GeminiModelClient(model=generator_model, api_key=api_key)),
            judge=StructuredGenerationJudge(GeminiModelClient(model=judge_model, api_key=api_key)),
            cache=JudgeCache(cache_path),
            generator_model=generator_model,
            judge_model=judge_model,
            configuration=configuration,
        )
        write_generation_results(result, arguments.output, arguments.report)
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(f"Wrote {arguments.output} and {arguments.report}")


if __name__ == "__main__":
    main()
