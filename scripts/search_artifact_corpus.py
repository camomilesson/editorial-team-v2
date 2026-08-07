#!/usr/bin/env python3
"""Search the local editorial artifact corpus with the production retriever."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO

from editorial_team.app.artifact_config import load_artifact_configuration
from editorial_team.app.retrieval_config import load_retrieval_configuration
from editorial_team.artifacts import HybridRetriever, ParagraphChunker, SQLiteArtifactStore
from editorial_team.artifacts.embeddings import SentenceTransformerEmbeddingModel
from editorial_team.artifacts.reranking import CrossEncoderReranker
from editorial_team.artifacts.retrieval_types import SearchResult
from editorial_team.contracts.common import parse_utc_timestamp, timestamp_to_json


def build_parser() -> argparse.ArgumentParser:
    """Return the reusable manual-search argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--created-from", type=_timestamp)
    parser.add_argument("--created-to", type=_timestamp)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prefer-recent", action="store_true")
    return parser


def render_results(results: Sequence[SearchResult], stream: TextIO) -> None:
    """Print exact ranks and stage diagnostics in a readable form."""

    if not results:
        print("No eligible chunks found.", file=stream)
        return
    for result in results:
        rerank = "none" if result.rerank_score is None else f"{result.rerank_score:.6f}"
        print(
            f"#{result.rank} artifact={result.artifact_id} chunk={result.chunk_id} "
            f"producer={result.producer.value} created={timestamp_to_json(result.created_at)}",
            file=stream,
        )
        print(
            f"   dense_rank={result.dense_rank} bm25_rank={result.bm25_rank} "
            f"rrf={result.rrf_score:.8f} reranker={rerank}",
            file=stream,
        )
        print(f"   {result.excerpt}", file=stream)


def execute_search(
    retriever: HybridRetriever,
    *,
    query: str,
    conversation_id: str,
    created_from: datetime | None,
    created_to: datetime | None,
    top_k: int,
    rerank: bool,
    prefer_recent: bool,
    stream: TextIO,
) -> tuple[SearchResult, ...]:
    """Run and render one search; injectable for deterministic CLI tests."""

    results = retriever.search(
        query=query,
        conversation_id=conversation_id,
        created_from=created_from,
        created_to=created_to,
        top_k=top_k,
        rerank=rerank,
        prefer_recent=prefer_recent,
    )
    render_results(results, stream)
    return results


def main() -> None:
    """Construct lazy local models and search the configured corpus."""

    import sys

    arguments = build_parser().parse_args()
    artifact_configuration = load_artifact_configuration()
    retrieval_configuration = load_retrieval_configuration()
    store = SQLiteArtifactStore(
        arguments.database or artifact_configuration.database_path,
        chunker=ParagraphChunker(),
    )
    store.initialize()
    try:
        retriever = HybridRetriever(
            store=store,
            embeddings=SentenceTransformerEmbeddingModel(
                retrieval_configuration.embedding_model
            ),
            reranker=CrossEncoderReranker(retrieval_configuration.reranker_model),
            dense_depth=retrieval_configuration.dense_depth,
            bm25_depth=retrieval_configuration.bm25_depth,
            rrf_k=retrieval_configuration.rrf_k,
            fused_depth=retrieval_configuration.fused_depth,
            rerank_depth=retrieval_configuration.rerank_depth,
        )
        execute_search(
            retriever,
            query=arguments.query,
            conversation_id=arguments.conversation_id,
            created_from=arguments.created_from,
            created_to=arguments.created_to,
            top_k=arguments.top_k or retrieval_configuration.top_k,
            rerank=(
                retrieval_configuration.rerank
                if arguments.rerank is None
                else arguments.rerank
            ),
            prefer_recent=arguments.prefer_recent,
            stream=sys.stdout,
        )
    finally:
        store.close()


def _timestamp(value: str) -> datetime:
    try:
        return parse_utc_timestamp(value, "timestamp")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 UTC timestamp") from exc


if __name__ == "__main__":
    main()
