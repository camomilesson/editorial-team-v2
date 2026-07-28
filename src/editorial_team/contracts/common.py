"""Shared validation and serialization helpers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def require_non_blank(value: str, field_name: str) -> str:
    """Return a trimmed value or reject an empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value.strip()


def require_utc_timestamp(value: datetime, field_name: str) -> datetime:
    """Require a timezone-aware timestamp normalized to UTC."""

    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if offset.total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC")
    return value


def parse_utc_timestamp(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp") from exc
    return require_utc_timestamp(parsed, field_name)


def timestamp_to_json(value: datetime) -> str:
    """Serialize a UTC timestamp using the explicit Z suffix."""

    require_utc_timestamp(value, "timestamp")
    return value.isoformat().replace("+00:00", "Z")


def require_json_object(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Require a JSON-compatible object."""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc
    return value
