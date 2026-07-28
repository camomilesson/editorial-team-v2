from datetime import UTC, datetime, timedelta, timezone

import pytest

from editorial_team.contracts.common import (
    parse_utc_timestamp,
    require_json_object,
    require_non_blank,
    require_utc_timestamp,
    timestamp_to_json,
)
from editorial_team.contracts.identity import validate_identifier


def test_common_contract_helpers_accept_and_normalize_valid_values() -> None:
    timestamp = datetime(2026, 7, 28, 10, 30, tzinfo=UTC)

    assert require_non_blank("  value  ", "field") == "value"
    assert require_utc_timestamp(timestamp, "created_at") is timestamp
    assert parse_utc_timestamp("2026-07-28T10:30:00Z", "created_at") == timestamp
    assert timestamp_to_json(timestamp) == "2026-07-28T10:30:00Z"
    assert require_json_object({"ok": [True]}, "payload") == {"ok": [True]}


@pytest.mark.parametrize(
    "operation",
    [
        lambda: require_non_blank(" ", "field"),
        lambda: require_utc_timestamp(datetime(2026, 1, 1), "created_at"),
        lambda: require_utc_timestamp(
            datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1))), "created_at"
        ),
        lambda: parse_utc_timestamp("not-a-date", "created_at"),
        lambda: require_json_object({"value": float("nan")}, "payload"),
    ],
)
def test_common_contract_helpers_reject_invalid_values(operation: object) -> None:
    with pytest.raises(ValueError):
        operation()  # type: ignore[operator]


@pytest.mark.parametrize("value", ["abc", "A_1", "id-with-dashes", "x" * 128])
def test_identifier_validation_accepts_opaque_values(value: str) -> None:
    assert validate_identifier(value, "identifier") == value


@pytest.mark.parametrize(
    "value",
    ["", " ", "../secret", "a..b", "/absolute", "has space", "x" * 129],
)
def test_identifier_validation_rejects_path_like_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        validate_identifier(value, "identifier")
