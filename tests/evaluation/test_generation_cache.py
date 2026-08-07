from pathlib import Path

from editorial_team.evaluation.generation_cache import JudgeCache, cache_key
from editorial_team.evaluation.generation_judges import JudgeScore


def test_cache_miss_store_hit_and_content_derived_invalidation(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = JudgeCache(path)
    base = {"case_id": "c", "answer": "a", "context": "x", "model": "m", "version": "v"}
    key = cache_key(**base)
    assert cache.get(key) is None
    cache.put(key, JudgeScore(1.0, "supported"))
    assert JudgeCache(path).get(key) == JudgeScore(1.0, "supported")
    for field in ("answer", "context", "model", "version"):
        changed = {**base, field: "changed"}
        assert cache_key(**changed) != key
    assert "credential" not in path.read_text()


def test_corrupt_cache_is_ignored_safely(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("not json")
    assert JudgeCache(path).get("missing") is None
