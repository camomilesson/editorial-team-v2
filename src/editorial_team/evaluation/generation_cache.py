"""Validated persistent content-addressed judge cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from editorial_team.evaluation.generation_judges import JudgeScore


class JudgeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self._values = self._load()

    def get(self, key: str) -> JudgeScore | None:
        value = self._values.get(key)
        try:
            score = None if value is None else JudgeScore(value["score"], value["reason"])
        except (KeyError, TypeError, ValueError):
            score = None
        if score is None:
            self.misses += 1
        else:
            self.hits += 1
        return score

    def put(self, key: str, score: JudgeScore) -> None:
        self._values[key] = {"score": score.score, "reason": score.reason}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._values, sort_keys=True), encoding="utf-8")

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def cache_key(**parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
