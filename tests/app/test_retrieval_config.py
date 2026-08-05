"""Local hybrid-retrieval configuration tests."""

import pytest

from editorial_team.app.retrieval_config import (
    DEFAULT_BM25_DEPTH,
    DEFAULT_DENSE_DEPTH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FUSED_DEPTH,
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RRF_K,
    RetrievalConfigurationError,
    load_retrieval_configuration,
)

VARIABLES = (
    "EDITORIAL_EMBEDDING_MODEL",
    "EDITORIAL_RERANKER_MODEL",
    "EDITORIAL_RETRIEVAL_DENSE_DEPTH",
    "EDITORIAL_RETRIEVAL_BM25_DEPTH",
    "EDITORIAL_RETRIEVAL_RRF_K",
    "EDITORIAL_RETRIEVAL_FUSED_DEPTH",
    "EDITORIAL_RETRIEVAL_RERANK_DEPTH",
    "EDITORIAL_RETRIEVAL_TOP_K",
)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    value = load_retrieval_configuration()
    assert value.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert value.reranker_model == DEFAULT_RERANKER_MODEL
    assert (
        value.dense_depth,
        value.bm25_depth,
        value.rrf_k,
        value.fused_depth,
        value.rerank_depth,
        value.top_k,
    ) == (
        DEFAULT_DENSE_DEPTH,
        DEFAULT_BM25_DEPTH,
        DEFAULT_RRF_K,
        DEFAULT_FUSED_DEPTH,
        DEFAULT_RERANK_DEPTH,
        DEFAULT_RETRIEVAL_TOP_K,
    )


def test_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITORIAL_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("EDITORIAL_RERANKER_MODEL", "reranker-test")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_DENSE_DEPTH", "20")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_BM25_DEPTH", "21")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_RRF_K", "40")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_FUSED_DEPTH", "12")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_RERANK_DEPTH", "8")
    monkeypatch.setenv("EDITORIAL_RETRIEVAL_TOP_K", "3")
    value = load_retrieval_configuration()
    assert value.embedding_model == "embedding-test"
    assert value.reranker_model == "reranker-test"
    assert value.dense_depth == 20
    assert value.bm25_depth == 21
    assert value.rrf_k == 40
    assert (value.fused_depth, value.rerank_depth, value.top_k) == (12, 8, 3)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EDITORIAL_EMBEDDING_MODEL", " "),
        ("EDITORIAL_RERANKER_MODEL", " "),
        ("EDITORIAL_RETRIEVAL_DENSE_DEPTH", "0"),
        ("EDITORIAL_RETRIEVAL_BM25_DEPTH", "-1"),
        ("EDITORIAL_RETRIEVAL_RRF_K", "invalid"),
        ("EDITORIAL_RETRIEVAL_TOP_K", "11"),
        ("EDITORIAL_RETRIEVAL_RERANK_DEPTH", "4"),
        ("EDITORIAL_RETRIEVAL_FUSED_DEPTH", "10"),
    ],
)
def test_invalid_settings_are_sanitized(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, value)
    if name == "EDITORIAL_RETRIEVAL_FUSED_DEPTH":
        monkeypatch.setenv("EDITORIAL_RETRIEVAL_RERANK_DEPTH", "15")
    with pytest.raises(RetrievalConfigurationError, match="configuration is invalid"):
        load_retrieval_configuration()
