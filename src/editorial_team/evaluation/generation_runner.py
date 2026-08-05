"""Standalone judged RAG generation evaluation over production retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from editorial_team.app.retrieval_config import RetrievalConfiguration
from editorial_team.artifacts import ParagraphChunker, SQLiteArtifactStore
from editorial_team.artifacts.embeddings import EmbeddingModel
from editorial_team.artifacts.reranking import Reranker
from editorial_team.artifacts.retrieval import HybridRetriever
from editorial_team.evaluation.generation_cache import JudgeCache, cache_key
from editorial_team.evaluation.generation_cases import (
    GenerationCase,
    resolve_generation_anchors,
)
from editorial_team.evaluation.generation_judges import (
    JUDGE_PROMPT_VERSION,
    SCORER_VERSION,
    GenerationMetric,
    StructuredGenerationJudge,
)
from editorial_team.evaluation.generation_models import (
    GENERATOR_PROMPT_VERSION,
    GenerationContext,
    GroundedGenerator,
)
from editorial_team.evaluation.retrieval_cases import CorpusArtifact, file_sha256
from editorial_team.evaluation.retrieval_runner import seed_corpus


def run_generation_evaluation(
    *,
    database_path: Path,
    corpus: tuple[CorpusArtifact, ...],
    cases: tuple[GenerationCase, ...],
    corpus_path: Path,
    cases_path: Path,
    embeddings: EmbeddingModel,
    reranker: Reranker,
    generator: GroundedGenerator,
    judge: StructuredGenerationJudge,
    cache: JudgeCache,
    generator_model: str,
    judge_model: str,
    configuration: RetrievalConfiguration,
    top_k: int = 5,
    run_at: datetime | None = None,
) -> dict[str, Any]:
    chunker = ParagraphChunker()
    golden_ids = resolve_generation_anchors(corpus, cases)
    all_chunks = {
        chunk.chunk_id: GenerationContext(chunk.chunk_id, chunk.artifact_id, chunk.content)
        for item in corpus
        for chunk in chunker.chunk(item.artifact)
    }
    store = SQLiteArtifactStore(database_path, chunker=chunker)
    seed_corpus(store, corpus)
    try:
        retriever = HybridRetriever(
            store=store,
            embeddings=embeddings,
            reranker=reranker,
            dense_depth=configuration.dense_depth,
            bm25_depth=configuration.bm25_depth,
            rrf_k=configuration.rrf_k,
            fused_depth=configuration.fused_depth,
            rerank_depth=configuration.rerank_depth,
        )
        conditions = {
            "on": _condition(
                cases,
                True,
                retriever,
                generator,
                judge,
                cache,
                golden_ids,
                all_chunks,
                top_k,
                corpus_path,
                cases_path,
                judge_model,
            ),
            "off": _condition(
                tuple(case for case in cases if case.compare_rerank_off),
                False,
                retriever,
                generator,
                judge,
                cache,
                golden_ids,
                all_chunks,
                top_k,
                corpus_path,
                cases_path,
                judge_model,
            ),
        }
    finally:
        store.close()
    return {
        "run_at": (run_at or datetime.now(UTC)).isoformat(),
        "corpus_sha256": file_sha256(corpus_path),
        "case_set_sha256": file_sha256(cases_path),
        "case_count": len(cases),
        "generator_model": generator_model,
        "judge_model": judge_model,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "scorer_version": SCORER_VERSION,
        "retrieval_configuration": {
            "embedding_model": embeddings.model_id,
            "reranker_model": getattr(reranker, "model_id", type(reranker).__name__),
            "dense_depth": configuration.dense_depth,
            "bm25_depth": configuration.bm25_depth,
            "rrf_k": configuration.rrf_k,
            "fused_depth": configuration.fused_depth,
            "rerank_depth": configuration.rerank_depth,
            "top_k": top_k,
        },
        "conditions": conditions,
        "cache": {"hits": cache.hits, "misses": cache.misses},
        "comparison": _comparison(conditions),
    }


def _condition(
    cases: tuple[GenerationCase, ...],
    rerank: bool,
    retriever: HybridRetriever,
    generator: GroundedGenerator,
    judge: StructuredGenerationJudge,
    cache: JudgeCache,
    golden_ids: dict[str, tuple[str, ...]],
    all_chunks: dict[str, GenerationContext],
    top_k: int,
    corpus_path: Path,
    cases_path: Path,
    judge_model: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        results = retriever.search(
            query=case.query,
            conversation_id="eval-retrieval-main",
            top_k=top_k,
            rerank=rerank,
        )
        contexts = tuple(
            GenerationContext(item.chunk_id, item.artifact_id, item.excerpt) for item in results
        )
        golden_contexts = tuple(all_chunks[item] for item in golden_ids[case.case_id])
        answer = generator.generate(case.query, contexts)
        scores = {}
        for metric in GenerationMetric:
            key = cache_key(
                case_id=case.case_id,
                metric=metric.value,
                case_set_hash=file_sha256(cases_path),
                corpus_hash=file_sha256(corpus_path),
                query_hash=_hash(case.query),
                candidate_answer_hash=_hash(answer),
                retrieved_context_hash=_hash(tuple(asdict(item) for item in contexts)),
                golden_answer_hash=_hash(case.golden_answer),
                golden_context_hash=_hash(tuple(asdict(item) for item in golden_contexts)),
                judge_model=judge_model,
                judge_prompt_version=JUDGE_PROMPT_VERSION,
                scorer_version=SCORER_VERSION,
            )
            score = cache.get(key)
            if score is None:
                score = judge.judge(
                    metric,
                    query=case.query,
                    candidate_answer=answer,
                    golden_answer=case.golden_answer,
                    retrieved_contexts=contexts,
                    golden_contexts=golden_contexts,
                )
                cache.put(key, score)
            scores[metric.value] = asdict(score)
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "failure_category": case.failure_category,
                "tags": list(case.tags),
                "expected_out_of_corpus": case.expected_out_of_corpus,
                "retrieved_chunk_ids": [item.chunk_id for item in contexts],
                "golden_chunk_ids": list(golden_ids[case.case_id]),
                "candidate_answer": answer,
                "golden_answer": case.golden_answer,
                "metrics": scores,
            }
        )
    return {
        "rerank": rerank,
        "case_ids": [case.case_id for case in cases],
        "cases": rows,
        "overall": _averages(rows),
        "categories": {
            category: _averages(tuple(row for row in rows if row["failure_category"] == category))
            for category in sorted({row["failure_category"] for row in rows})
        },
    }


def _averages(rows: Any) -> dict[str, float]:
    return {
        metric.value: fmean(row["metrics"][metric.value]["score"] for row in rows)
        for metric in GenerationMetric
    }


def _comparison(conditions: dict[str, Any]) -> dict[str, Any]:
    on = {row["case_id"]: row for row in conditions["on"]["cases"]}
    output = {"improved": [], "unchanged": [], "worsened": [], "per_case": {}}
    for row in conditions["off"]["cases"]:
        case_id = row["case_id"]
        delta = fmean(row["metrics"][metric.value]["score"] for metric in GenerationMetric) - fmean(
            on[case_id]["metrics"][metric.value]["score"] for metric in GenerationMetric
        )
        status = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
        output[status].append(case_id)
        output["per_case"][case_id] = delta
    return output


def _hash(value: Any) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()
