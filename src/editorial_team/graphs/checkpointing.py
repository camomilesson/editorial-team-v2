"""Durable SQLite checkpoint construction."""

from __future__ import annotations

import math
import os
import pickle
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


class _LocalDomainSerializer:
    """Serialize trusted local immutable domain objects for SQLite checkpoints."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_name, payload = data
        if type_name != "pickle":
            raise ValueError("Unsupported local checkpoint payload")
        return pickle.loads(payload)


def create_sqlite_checkpointer(
    path: Path,
    *,
    busy_timeout_seconds: float = 5.0,
) -> tuple[SqliteSaver, Callable[[], None]]:
    """Create and initialize one application-lifetime SQLite checkpointer."""

    if not isinstance(path, Path) or not str(path).strip():
        raise ValueError("Checkpoint database path is invalid")
    if (
        isinstance(busy_timeout_seconds, bool)
        or not isinstance(busy_timeout_seconds, (int, float))
        or not math.isfinite(busy_timeout_seconds)
        or busy_timeout_seconds < 0
    ):
        raise ValueError("Checkpoint busy timeout is invalid")
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix" and not parent_existed:
        path.parent.chmod(0o700)
    database_existed = path.exists()
    connection = sqlite3.connect(
        path,
        timeout=float(busy_timeout_seconds),
        check_same_thread=False,
    )
    if os.name == "posix" and not database_existed:
        path.chmod(0o600)
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
    saver = SqliteSaver(connection, serde=_LocalDomainSerializer())
    try:
        saver.setup()
    except Exception:
        connection.close()
        raise
    return saver, connection.close
