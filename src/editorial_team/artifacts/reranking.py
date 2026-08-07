"""Reranking contracts and lazy local cross-encoder adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    """Score query/passage pairs for relevance."""

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Lazy local Sentence Transformers cross-encoder reranker."""

    def __init__(self, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must not be blank")
        self.model_id = model_id.strip()
        self._model: Any | None = None

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        if not passages:
            return ()
        pairs = [(query, passage) for passage in passages]
        values = self._load().predict(pairs)
        return [float(value) for value in values]

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id)
        return self._model
