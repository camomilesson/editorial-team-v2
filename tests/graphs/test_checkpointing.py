"""Tests for durable SQLite checkpoint construction."""

import os
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from editorial_team.graphs import create_sqlite_checkpointer


def test_sqlite_checkpointer_initializes_and_closes(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "conversations.db"
    saver, close = create_sqlite_checkpointer(database)
    assert isinstance(saver, SqliteSaver)
    assert database.exists()
    if os.name == "posix":
        assert database.stat().st_mode & 0o777 == 0o600
        assert database.parent.stat().st_mode & 0o777 == 0o700
    close()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_sqlite_checkpointer_rejects_invalid_busy_timeout(
    tmp_path: Path, timeout: object
) -> None:
    with pytest.raises(ValueError, match="busy timeout"):
        create_sqlite_checkpointer(
            tmp_path / "state.db",
            busy_timeout_seconds=timeout,  # type: ignore[arg-type]
        )
