"""Paragraph-aware artifact chunking tests."""

from datetime import UTC, datetime

from editorial_team.artifacts import (
    ArtifactProducer,
    EditorialArtifact,
    ParagraphChunker,
    artifact_id_for,
    content_sha256,
)


def artifact(content: str, task_id: str = "task-1") -> EditorialArtifact:
    return EditorialArtifact(
        artifact_id=artifact_id_for(task_id, ArtifactProducer.WRITER),
        task_id=task_id,
        producer=ArtifactProducer.WRITER,
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
        conversation_id="conversation-1",
        user_request="Write it",
        content=content,
        content_sha256=content_sha256(content),
    )


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_short_draft_remains_one_exact_chunk() -> None:
    value = artifact("# Heading\n\nA short paragraph.\n\nAnother paragraph.")
    chunks = ParagraphChunker(target_tokens=20, max_tokens=30, overlap_tokens=3).chunk(value)
    assert len(chunks) == 1
    assert chunks[0].content == value.content
    assert (chunks[0].character_start, chunks[0].character_end) == (0, len(value.content))


def test_paragraph_boundaries_heading_and_offsets_are_preserved() -> None:
    content = f"# Heading\n\n{words('a', 8)}\n\n{words('b', 8)}\n\n{words('c', 8)}"
    chunks = ParagraphChunker(target_tokens=12, max_tokens=18, overlap_tokens=3).chunk(
        artifact(content)
    )
    assert chunks[0].content.startswith("# Heading\n\n")
    assert "a0" in chunks[0].content
    assert all(
        content[chunk.character_start : chunk.character_end] == chunk.content for chunk in chunks
    )
    assert all(not chunk.content.rstrip().endswith("Heading") for chunk in chunks)


def test_oversized_paragraph_uses_deterministic_hard_fallback() -> None:
    value = artifact(words("word", 35))
    chunker = ParagraphChunker(target_tokens=10, max_tokens=12, overlap_tokens=2)
    first = chunker.chunk(value)
    second = chunker.chunk(value)
    assert first == second
    assert len(first) == 3
    assert all(len(chunk.content.split()) <= 12 for chunk in first)
    assert all(
        value.content[chunk.character_start : chunk.character_end] == chunk.content
        for chunk in first
    )


def test_overlap_reuses_complete_trailing_paragraphs() -> None:
    content = "\n\n".join(words(prefix, 4) for prefix in ("a", "b", "c", "d"))
    chunks = ParagraphChunker(target_tokens=8, max_tokens=12, overlap_tokens=4).chunk(
        artifact(content)
    )
    assert len(chunks) >= 2
    assert words("b", 4) in chunks[0].content
    assert words("b", 4) in chunks[1].content


def test_chunk_ids_include_artifact_and_chunker_identity() -> None:
    value = artifact(words("word", 25))
    first = ParagraphChunker(target_tokens=10, max_tokens=12, overlap_tokens=2).chunk(value)
    other_artifact = artifact(value.content, "task-2")
    second = ParagraphChunker(target_tokens=10, max_tokens=12, overlap_tokens=2).chunk(
        other_artifact
    )
    third = ParagraphChunker(
        target_tokens=10,
        max_tokens=12,
        overlap_tokens=2,
        version="paragraph-heading-v2",
    ).chunk(value)
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in third]
