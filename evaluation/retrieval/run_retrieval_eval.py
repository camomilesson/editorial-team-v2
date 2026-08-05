#!/usr/bin/env python3
"""Run the fixed HW2 retrieval evaluation with production local models."""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from editorial_team.app.retrieval_config import load_retrieval_configuration
from editorial_team.artifacts.embeddings import SentenceTransformerEmbeddingModel
from editorial_team.artifacts.reranking import CrossEncoderReranker
from editorial_team.evaluation.retrieval_cases import load_cases, load_corpus
from editorial_team.evaluation.retrieval_reporting import write_results
from editorial_team.evaluation.retrieval_runner import run_evaluation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path(__file__).with_name("corpus.json")
DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_JSON = ROOT / "evaluation/outputs/retrieval_results.json"
DEFAULT_REPORT = ROOT / "evaluation/outputs/retrieval_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rerank", choices=("both", "on", "off"), default="both")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    arguments = parser.parse_args()
    configuration = load_retrieval_configuration()
    modes = {"both": (False, True), "on": (True,), "off": (False,)}[arguments.rerank]
    temporary: tempfile.TemporaryDirectory[str] | None = None
    database = arguments.database
    if database is None:
        temporary = tempfile.TemporaryDirectory(prefix="editorial-retrieval-eval-")
        database = Path(temporary.name) / "evaluation.db"
    try:
        result = run_evaluation(
            database_path=database,
            corpus=load_corpus(DEFAULT_CORPUS),
            cases=load_cases(DEFAULT_CASES),
            corpus_path=DEFAULT_CORPUS,
            cases_path=DEFAULT_CASES,
            embeddings=SentenceTransformerEmbeddingModel(configuration.embedding_model),
            reranker=CrossEncoderReranker(configuration.reranker_model),
            configuration=configuration,
            k_values=tuple(arguments.k),
            rerank_modes=modes,
            run_at=datetime.now(UTC),
        )
        write_results(result, arguments.output, arguments.report)
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(f"Wrote {arguments.output} and {arguments.report}")


if __name__ == "__main__":
    main()
