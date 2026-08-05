"""Sanitized environment configuration for optional live heartbeats."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 900.0
DEFAULT_HEARTBEAT_DB_PATH = Path("runtime_data/editorial_team.db")
MIN_LIVE_HEARTBEAT_INTERVAL_SECONDS = 10.0


class HeartbeatConfigurationError(RuntimeError):
    """Optional heartbeat configuration is invalid."""


@dataclass(frozen=True)
class HeartbeatConfiguration:
    """Validated non-secret live heartbeat settings."""

    enabled: bool = False
    interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    database_path: Path = DEFAULT_HEARTBEAT_DB_PATH
    maintainer_chat_id: int | None = None


def load_heartbeat_configuration() -> HeartbeatConfiguration:
    """Load optional heartbeat settings without exposing raw values."""

    enabled = _boolean(os.getenv("EDITORIAL_HEARTBEAT_ENABLED"), default=False)
    interval = _interval(
        os.getenv("EDITORIAL_HEARTBEAT_INTERVAL_SECONDS"),
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )
    raw_path = os.getenv(
        "EDITORIAL_HEARTBEAT_DB_PATH",
        str(DEFAULT_HEARTBEAT_DB_PATH),
    )
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HeartbeatConfigurationError("Heartbeat database path is invalid")
    raw_chat_id = os.getenv("EDITORIAL_ADMIN_TELEGRAM_CHAT_ID")
    chat_id = _chat_id(raw_chat_id) if raw_chat_id and raw_chat_id.strip() else None
    if enabled and chat_id is None:
        raise HeartbeatConfigurationError("Enabled heartbeat requires maintainer configuration")
    return HeartbeatConfiguration(
        enabled=enabled,
        interval_seconds=interval,
        database_path=Path(raw_path.strip()),
        maintainer_chat_id=chat_id,
    )


def _boolean(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise HeartbeatConfigurationError("Heartbeat enabled setting is invalid")


def _interval(value: str | None, *, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        interval = float(value)
    except (TypeError, ValueError):
        raise HeartbeatConfigurationError("Heartbeat interval is invalid") from None
    if not math.isfinite(interval) or interval < MIN_LIVE_HEARTBEAT_INTERVAL_SECONDS:
        raise HeartbeatConfigurationError("Heartbeat interval is invalid")
    return interval


def _chat_id(value: str) -> int:
    try:
        chat_id = int(value)
    except (TypeError, ValueError):
        raise HeartbeatConfigurationError("Maintainer configuration is invalid") from None
    if not value.strip() or chat_id == 0 or str(chat_id) != value.strip():
        raise HeartbeatConfigurationError("Maintainer configuration is invalid")
    return chat_id
