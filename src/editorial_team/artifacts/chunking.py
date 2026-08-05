"""Deterministic paragraph- and heading-aware artifact chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from editorial_team.artifacts.models import ArtifactChunk, EditorialArtifact, content_sha256
from editorial_team.contracts.identity import validate_identifier

DEFAULT_TARGET_TOKENS = 700
DEFAULT_MAX_TOKENS = 1000
DEFAULT_OVERLAP_TOKENS = 90
DEFAULT_CHUNKER_VERSION = "paragraph-heading-v1"

_TOKEN = re.compile(r"\S+")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s+|$)")
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


class ParagraphChunker:
    """Create stable chunks while preserving exact source offsets."""

    def __init__(
        self,
        *,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        version: str = DEFAULT_CHUNKER_VERSION,
    ) -> None:
        for name, value in (
            ("target_tokens", target_tokens),
            ("max_tokens", max_tokens),
            ("overlap_tokens", overlap_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if target_tokens <= 0 or max_tokens < target_tokens or overlap_tokens >= target_tokens:
            raise ValueError("chunk token limits are inconsistent")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.version = validate_identifier(version, "version")

    def chunk(self, artifact: EditorialArtifact) -> tuple[ArtifactChunk, ...]:
        """Return deterministic chunks for one complete artifact."""

        if not isinstance(artifact, EditorialArtifact):
            raise ValueError("artifact must be an EditorialArtifact")
        text = artifact.content
        if self._tokens(text) <= self.max_tokens:
            spans = (_Span(0, len(text)),)
        else:
            units = self._units(text)
            spans = self._pack(text, units)
        return tuple(
            self._chunk_from_span(artifact, ordinal, span) for ordinal, span in enumerate(spans)
        )

    def _units(self, text: str) -> tuple[_Span, ...]:
        paragraphs = self._paragraphs(text)
        expanded: list[_Span] = []
        for paragraph in paragraphs:
            if self._tokens(text[paragraph.start : paragraph.end]) <= self.max_tokens:
                expanded.append(paragraph)
            else:
                expanded.extend(self._split_oversized(text, paragraph))

        grouped: list[_Span] = []
        index = 0
        while index < len(expanded):
            current = expanded[index]
            current_text = text[current.start : current.end].strip()
            if (
                _HEADING.match(current_text)
                and index + 1 < len(expanded)
                and self._tokens(text[current.start : expanded[index + 1].end]) <= self.max_tokens
            ):
                grouped.append(_Span(current.start, expanded[index + 1].end))
                index += 2
            else:
                grouped.append(current)
                index += 1
        return tuple(grouped)

    @staticmethod
    def _paragraphs(text: str) -> tuple[_Span, ...]:
        spans: list[_Span] = []
        start = 0
        for match in re.finditer(r"(?:\r?\n[ \t]*){2,}", text):
            end = match.start()
            if text[start:end].strip():
                spans.append(_Span(start, end))
            start = match.end()
        if text[start:].strip():
            spans.append(_Span(start, len(text)))
        return tuple(spans)

    def _split_oversized(self, text: str, span: _Span) -> tuple[_Span, ...]:
        local = text[span.start : span.end]
        boundaries = [0]
        boundaries.extend(match.end() for match in _SENTENCE_END.finditer(local))
        boundaries.append(len(local))
        pieces: list[_Span] = []
        piece_start = boundaries[0]
        for boundary in boundaries[1:]:
            candidate = local[piece_start:boundary]
            if self._tokens(candidate) > self.max_tokens:
                if (
                    boundary != boundaries[-1]
                    and piece_start != boundaries[boundaries.index(boundary) - 1]
                ):
                    previous = boundaries[boundaries.index(boundary) - 1]
                    if previous > piece_start:
                        pieces.append(_Span(span.start + piece_start, span.start + previous))
                        piece_start = previous
                        candidate = local[piece_start:boundary]
                if self._tokens(candidate) > self.max_tokens:
                    pieces.extend(self._hard_slices(local, span.start, piece_start, boundary))
                    piece_start = boundary
            elif boundary == boundaries[-1] and candidate.strip():
                pieces.append(_Span(span.start + piece_start, span.start + boundary))
        return tuple(piece for piece in pieces if text[piece.start : piece.end].strip())

    def _hard_slices(self, local: str, base: int, start: int, end: int) -> tuple[_Span, ...]:
        matches = list(_TOKEN.finditer(local, start, end))
        if not matches:
            return ()
        spans: list[_Span] = []
        offset = 0
        while offset < len(matches):
            batch = matches[offset : offset + self.max_tokens]
            batch_start = start if offset == 0 else batch[0].start()
            batch_end = end if offset + self.max_tokens >= len(matches) else batch[-1].end()
            spans.append(_Span(base + batch_start, base + batch_end))
            offset += self.max_tokens
        return tuple(spans)

    def _pack(self, text: str, units: tuple[_Span, ...]) -> tuple[_Span, ...]:
        chunks: list[_Span] = []
        start_index = 0
        while start_index < len(units):
            end_index = start_index
            while end_index + 1 < len(units):
                candidate = text[units[start_index].start : units[end_index + 1].end]
                if self._tokens(candidate) > self.target_tokens:
                    break
                end_index += 1
            chunks.append(_Span(units[start_index].start, units[end_index].end))
            if end_index == len(units) - 1:
                break
            overlap_start = end_index
            overlap_count = 0
            while overlap_start >= start_index:
                count = self._tokens(text[units[overlap_start].start : units[end_index].end])
                if count > self.overlap_tokens:
                    break
                overlap_count = count
                overlap_start -= 1
            next_start = overlap_start + 1 if overlap_count else end_index + 1
            if next_start <= start_index:
                next_start = end_index + 1
            start_index = next_start
        return tuple(chunks)

    def _chunk_from_span(
        self, artifact: EditorialArtifact, ordinal: int, span: _Span
    ) -> ArtifactChunk:
        value = artifact.content[span.start : span.end]
        digest = content_sha256(value)
        normalized = "\n".join(
            line.rstrip()
            for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ).strip()
        normalized_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        identity = (
            f"chunk:v1:{artifact.artifact_id}:{self.version}:{ordinal}:{normalized_digest}"
        )
        chunk_id = f"chunk-v1-{hashlib.sha256(identity.encode()).hexdigest()}"
        return ArtifactChunk(
            chunk_id=chunk_id,
            artifact_id=artifact.artifact_id,
            ordinal=ordinal,
            content=value,
            character_start=span.start,
            character_end=span.end,
            created_at=artifact.created_at,
            producer=artifact.producer,
            conversation_id=artifact.conversation_id,
            chunker_version=self.version,
            content_sha256=digest,
        )

    @staticmethod
    def _tokens(text: str) -> int:
        return len(_TOKEN.findall(text))
