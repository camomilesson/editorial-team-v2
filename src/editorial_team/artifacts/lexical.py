"""Deterministic Unicode-aware BM25 retrieval over artifact chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from editorial_team.artifacts.models import SearchableArtifactChunk

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Case-fold Unicode alphanumerics and discard punctuation separators."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return tuple(match.group(0).casefold() for match in _TOKEN.finditer(text))


@dataclass(frozen=True)
class LexicalCandidate:
    chunk: SearchableArtifactChunk
    rank: int
    score: float


class BM25Retriever:
    """Calculate local Okapi BM25 scores over one already-filtered corpus."""

    def __init__(self, *, depth: int, k1: float = BM25_K1, b: float = BM25_B) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        if not isinstance(k1, (int, float)) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not isinstance(b, (int, float)) or not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.depth = depth
        self.k1 = float(k1)
        self.b = float(b)

    def rank(
        self, query: str, chunks: Sequence[SearchableArtifactChunk]
    ) -> tuple[LexicalCandidate, ...]:
        """Return 1-based BM25 candidates with stable chunk-ID tie-breaking."""

        if not chunks:
            return ()
        query_terms = tokenize(query)
        documents = [tokenize(item.chunk.content) for item in chunks]
        average_length = sum(len(document) for document in documents) / len(documents)
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        scores: list[tuple[SearchableArtifactChunk, float]] = []
        for item, document in zip(chunks, documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                matching = document_frequency[term]
                inverse_frequency = math.log(
                    1 + (len(documents) - matching + 0.5) / (matching + 0.5)
                )
                length_ratio = 0.0 if average_length == 0 else len(document) / average_length
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += inverse_frequency * (frequency * (self.k1 + 1)) / denominator
            scores.append((item, score))
        ordered = sorted(scores, key=lambda value: (-value[1], value[0].chunk.chunk_id))[
            : self.depth
        ]
        return tuple(
            LexicalCandidate(chunk=item, rank=index, score=score)
            for index, (item, score) in enumerate(ordered, start=1)
        )
