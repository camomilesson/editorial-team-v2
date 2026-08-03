"""Configuration for durable conversation checkpoints."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT_DB_PATH = Path("runtime_data/conversations.db")
DEFAULT_CHECKPOINT_BUSY_TIMEOUT_SECONDS = 5.0


class CheckpointConfigurationError(RuntimeError):
    """Conversation checkpoint configuration is invalid."""


@dataclass(frozen=True)
class CheckpointConfiguration:
    """Validated non-secret checkpoint configuration."""

    database_path: Path = DEFAULT_CHECKPOINT_DB_PATH
    busy_timeout_seconds: float = DEFAULT_CHECKPOINT_BUSY_TIMEOUT_SECONDS


def load_checkpoint_configuration() -> CheckpointConfiguration:
    """Load the local checkpoint path without exposing invalid raw values."""

    raw_path = os.getenv("EDITORIAL_CHECKPOINT_DB_PATH", str(DEFAULT_CHECKPOINT_DB_PATH))
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise CheckpointConfigurationError("Checkpoint database path is invalid")
    raw_timeout = os.getenv(
        "EDITORIAL_CHECKPOINT_BUSY_TIMEOUT_SECONDS",
        str(DEFAULT_CHECKPOINT_BUSY_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        raise CheckpointConfigurationError("Checkpoint busy timeout is invalid") from None
    if not math.isfinite(timeout) or timeout < 0:
        raise CheckpointConfigurationError("Checkpoint busy timeout is invalid")
    return CheckpointConfiguration(Path(raw_path.strip()), timeout)
