"""Evaluation runner over the unmodified production hybrid retriever."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from editorial_team.app.retrieval_config import RetrievalConfiguration
from editorial_team.artifacts import ParagraphChunker, SQLiteArtifactStore
from editorial_team.artifacts.embeddings import EmbeddingModel
from editorial_team.artifacts.reranking import Reranker
from editorial_team.artifacts.retrieval import HybridRetriever, RetrievalStages
from editorial_team.artifacts.retrieval_types import SearchRequest
from editorial_team.evaluation.retrieval_cases import (
    CorpusArtifact,
    RetrievalCase,
    file_sha256,
    resolve_golden_chunks,
)
from editorial_team.evaluation.retrieval_metrics import aggregate_metrics, metrics_at_k


def seed_corpus(store: SQLiteArtifactStore, corpus: tuple[CorpusArtifact, ...]) -> None:
    store.initialize()
    runs: dict[str, list[CorpusArtifact]] = defaultdict(list)
    for item in corpus:
        runs[item.artifact.task_id].append(item)
    for task_id in sorted(runs):
        items = sorted(runs[task_id], key=lambda value: value.artifact.producer.value, reverse=True)
        store.save_run(tuple(item.artifact for item in items))


def run_evaluation(
    *,
    database_path: Path,
    corpus: tuple[CorpusArtifact, ...],
    cases: tuple[RetrievalCase, ...],
    corpus_path: Path,
    cases_path: Path,
    embeddings: EmbeddingModel,
    reranker: Reranker,
    configuration: RetrievalConfiguration,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    rerank_modes: tuple[bool, ...] = (False, True),
    run_at: datetime | None = None,
) -> dict[str, Any]:
    """Seed a dedicated database and measure exact production ranking sequences."""

    chunker = ParagraphChunker()
    golden = resolve_golden_chunks(corpus, cases, chunker)
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
            _mode(mode): _run_condition(retriever, cases, golden, k_values, mode)
            for mode in rerank_modes
        }
        chunk_count = sum(len(chunker.chunk(item.artifact)) for item in corpus)
        return {
            "configuration": {
                "run_at": (run_at or datetime.now(UTC)).isoformat(),
                "chunker_version": chunker.version,
                "target_tokens": chunker.target_tokens,
                "max_tokens": chunker.max_tokens,
                "overlap_tokens": chunker.overlap_tokens,
                "embedding_model": embeddings.model_id,
                "reranker_model": getattr(reranker, "model_id", type(reranker).__name__),
                "dense_depth": configuration.dense_depth,
                "bm25_depth": configuration.bm25_depth,
                "rrf_k": configuration.rrf_k,
                "fused_depth": configuration.fused_depth,
                "rerank_depth": configuration.rerank_depth,
                "k_values": list(k_values),
                "rerank_modes": [_mode(value) for value in rerank_modes],
            },
            "corpus": {
                "sha256": file_sha256(corpus_path),
                "artifact_count": len(corpus),
                "chunk_count": chunk_count,
                "conversation_counts": dict(
                    sorted(Counter(item.artifact.conversation_id for item in corpus).items())
                ),
            },
            "case_set": {
                "sha256": file_sha256(cases_path),
                "case_count": len(cases),
                "tag_counts": dict(
                    sorted(Counter(tag for case in cases for tag in case.tags).items())
                ),
            },
            "golden_chunk_ids": golden,
            "conditions": conditions,
            "reranking_deltas": _deltas(conditions, k_values),
        }
    finally:
        store.close()


def _run_condition(
    retriever: HybridRetriever,
    cases: tuple[RetrievalCase, ...],
    golden: dict[str, tuple[str, ...]],
    k_values: tuple[int, ...],
    rerank: bool,
) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    aggregate_input: list[tuple[str, tuple[str, ...], frozenset[str]]] = []
    for case in cases:
        stages = retriever.search_with_stages(
            SearchRequest(
                query=case.query,
                conversation_id=case.conversation_id,
                created_from=case.created_from,
                created_to=case.created_to,
                prefer_recent=case.prefer_recent,
                top_k=max(k_values),
                rerank=rerank,
            )
        )
        predictions = tuple(item.chunk_id for item in stages.results)
        relevant = frozenset(golden[case.case_id])
        aggregate_input.append((case.case_id, predictions, relevant))
        case_rows.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "query": case.query,
                "conversation_id": case.conversation_id,
                "created_from": None
                if case.created_from is None
                else case.created_from.isoformat(),
                "created_to": None if case.created_to is None else case.created_to.isoformat(),
                "prefer_recent": case.prefer_recent,
                "tags": list(case.tags),
                "empty_golden": case.empty_golden,
                "golden_chunk_ids": list(golden[case.case_id]),
                "stage_rankings": _rankings(stages),
                "final_rankings": list(predictions),
                "metrics": None
                if not relevant
                else {
                    str(k): asdict(metrics_at_k(predictions, relevant, k)) for k in k_values
                },
                "analysis": _analysis(stages, relevant),
            }
        )
    aggregates, empty_count = aggregate_metrics(tuple(aggregate_input), k_values)
    return {
        "rerank": rerank,
        "empty_golden_count": empty_count,
        "aggregate_metrics": {str(k): asdict(value) for k, value in aggregates.items()},
        "cases": case_rows,
    }


def _rankings(stages: RetrievalStages) -> dict[str, list[str]]:
    return {
        "dense": [item.chunk.chunk.chunk_id for item in stages.dense],
        "bm25": [item.chunk.chunk.chunk_id for item in stages.bm25],
        "rrf": [item.chunk.chunk.chunk_id for item in stages.fused],
        "final": [item.chunk_id for item in stages.results],
    }


def _analysis(stages: RetrievalStages, golden: frozenset[str]) -> dict[str, Any]:
    rankings = _rankings(stages)
    positions = {
        name: {chunk_id: index for index, chunk_id in enumerate(values, 1)}
        for name, values in rankings.items()
    }
    dense_found = golden.intersection(rankings["dense"])
    bm25_found = golden.intersection(rankings["bm25"])
    rrf_found = golden.intersection(rankings["rrf"])
    findings: list[str] = []
    best = {
        name: min((mapping[item] for item in golden if item in mapping), default=None)
        for name, mapping in positions.items()
    }
    if best["dense"] is not None and (
        best["bm25"] is None or best["dense"] < best["bm25"]
    ):
        findings.append("Dense ranked the best relevant evidence above BM25.")
    if best["bm25"] is not None and (
        best["dense"] is None or best["bm25"] < best["dense"]
    ):
        findings.append("BM25 ranked the best relevant evidence above dense search.")
    if rrf_found and bool(dense_found) != bool(bm25_found):
        findings.append("RRF retained evidence supplied strongly by only one retrieval branch.")
    if golden and not rrf_found:
        findings.append("No retrieval stage recovered a relevant chunk.")
    if golden and rrf_found and not findings:
        findings.append(
            "Both branches found the evidence; fusion preserved it without a unique fix."
        )
    if best["rrf"] is not None and best["final"] is not None:
        if best["final"] < best["rrf"]:
            findings.append("Reranking promoted the first relevant chunk.")
        elif best["final"] > best["rrf"]:
            findings.append("Reranking demoted the first relevant chunk.")
    if not golden:
        findings.append("Out-of-corpus qualitative probe; deterministic metrics are N/A.")
    return {
        "relevant_positions": {
            name: {item: mapping.get(item) for item in sorted(golden)}
            for name, mapping in positions.items()
        },
        "dense_found": sorted(dense_found),
        "bm25_found": sorted(bm25_found),
        "rrf_found": sorted(rrf_found),
        "final_found": sorted(golden.intersection(rankings["final"])),
        "findings": findings,
    }


def _deltas(conditions: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    if "off" not in conditions or "on" not in conditions:
        return {}
    output: dict[str, Any] = {"aggregate": {}, "cases": {}}
    for k in k_values:
        off = conditions["off"]["aggregate_metrics"][str(k)]
        on = conditions["on"]["aggregate_metrics"][str(k)]
        output["aggregate"][str(k)] = {key: on[key] - off[key] for key in off}
    off_cases = {row["case_id"]: row for row in conditions["off"]["cases"]}
    for row in conditions["on"]["cases"]:
        before = off_cases[row["case_id"]]
        metrics = {
            str(k): None
            if row["metrics"] is None
            else {
                key: row["metrics"][str(k)][key] - before["metrics"][str(k)][key]
                for key in row["metrics"][str(k)]
            }
            for k in k_values
        }
        maximum_metrics = metrics[str(max(k_values))]
        ndcg_delta = None if maximum_metrics is None else maximum_metrics["ndcg"]
        status = (
            "not_applicable"
            if ndcg_delta is None
            else "improved"
            if ndcg_delta > 0
            else "worsened"
            if ndcg_delta < 0
            else "unchanged"
        )
        output["cases"][row["case_id"]] = {"metrics": metrics, "status": status}
    return output


def _mode(value: bool) -> str:
    return "on" if value else "off"
