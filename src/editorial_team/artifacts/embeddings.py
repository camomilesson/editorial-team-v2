"""Embedding contracts and lazy local Sentence Transformers adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingModel(Protocol):
    """Embed corpus passages and one search query."""

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class SentenceTransformerEmbeddingModel:
    """Lazy local Sentence Transformers embedding model."""

    def __init__(self, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must not be blank")
        self._model_id = model_id.strip()
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        vectors = self._load().encode(list(texts), convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> Sequence[float]:
        vector = self._load().encode(text, convert_to_numpy=True)
        return vector.tolist()

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id)
        return self._model
