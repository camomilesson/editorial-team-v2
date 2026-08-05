"""Configuration for local hybrid editorial-artifact retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_DENSE_DEPTH = 30
DEFAULT_BM25_DEPTH = 30
DEFAULT_RRF_K = 60
DEFAULT_FUSED_DEPTH = 30
DEFAULT_RERANK_DEPTH = 15
DEFAULT_RETRIEVAL_TOP_K = 5
MAX_RETRIEVAL_TOP_K = 10


class RetrievalConfigurationError(RuntimeError):
    """Hybrid-retrieval configuration is invalid."""


@dataclass(frozen=True)
class RetrievalConfiguration:
    """Validated local model identifiers and retrieval depths."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    dense_depth: int = DEFAULT_DENSE_DEPTH
    bm25_depth: int = DEFAULT_BM25_DEPTH
    rrf_k: int = DEFAULT_RRF_K
    fused_depth: int = DEFAULT_FUSED_DEPTH
    rerank_depth: int = DEFAULT_RERANK_DEPTH
    top_k: int = DEFAULT_RETRIEVAL_TOP_K

    def __post_init__(self) -> None:
        for field_name in ("embedding_model", "reranker_model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value.strip())
        for field_name in (
            "dense_depth",
            "bm25_depth",
            "rrf_k",
            "fused_depth",
            "rerank_depth",
            "top_k",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.top_k > MAX_RETRIEVAL_TOP_K:
            raise ValueError(f"top_k must not exceed {MAX_RETRIEVAL_TOP_K}")
        if not self.top_k <= self.rerank_depth <= self.fused_depth:
            raise ValueError("retrieval depths must satisfy top_k <= rerank_depth <= fused_depth")


def load_retrieval_configuration() -> RetrievalConfiguration:
    """Load and sanitize local hybrid-retrieval settings."""

    try:
        return RetrievalConfiguration(
            embedding_model=_text("EDITORIAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            reranker_model=_text("EDITORIAL_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            dense_depth=_integer("EDITORIAL_RETRIEVAL_DENSE_DEPTH", DEFAULT_DENSE_DEPTH),
            bm25_depth=_integer("EDITORIAL_RETRIEVAL_BM25_DEPTH", DEFAULT_BM25_DEPTH),
            rrf_k=_integer("EDITORIAL_RETRIEVAL_RRF_K", DEFAULT_RRF_K),
            fused_depth=_integer("EDITORIAL_RETRIEVAL_FUSED_DEPTH", DEFAULT_FUSED_DEPTH),
            rerank_depth=_integer("EDITORIAL_RETRIEVAL_RERANK_DEPTH", DEFAULT_RERANK_DEPTH),
            top_k=_integer("EDITORIAL_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K),
        )
    except (TypeError, ValueError):
        raise RetrievalConfigurationError("Retrieval configuration is invalid") from None


def _text(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not value.strip():
        raise ValueError("blank setting")
    return value.strip()


def _integer(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)
