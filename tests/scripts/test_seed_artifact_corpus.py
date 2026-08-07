"""Development artifact seed utility tests."""

from pathlib import Path

from editorial_team.artifacts import ParagraphChunker, SQLiteArtifactStore
from scripts.seed_artifact_corpus import DEFAULT_FIXTURE_PATH, seed


def test_seed_uses_real_store_and_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "seed.db"
    assert seed(database, DEFAULT_FIXTURE_PATH) == 2
    assert seed(database, DEFAULT_FIXTURE_PATH) == 2
    store = SQLiteArtifactStore(database, chunker=ParagraphChunker())
    store.initialize()
    artifacts = store.list_artifacts()
    assert len(artifacts) == 2
    assert {artifact.producer.value for artifact in artifacts} == {"writer", "editor"}
    assert all(store.get_chunks(artifact.artifact_id) for artifact in artifacts)
    store.close()
