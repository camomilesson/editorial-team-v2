"""Validation for opaque identifiers."""

from __future__ import annotations

import re
from typing import NewType

from editorial_team.contracts.common import require_non_blank

Identifier = NewType("Identifier", str)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def validate_identifier(value: str, field_name: str) -> str:
    """Reject missing, path-like, or traversal-capable identifiers."""

    value = require_non_blank(value, field_name)
    if not _ID_PATTERN.fullmatch(value) or ".." in value:
        raise ValueError(f"{field_name} must be an opaque identifier without path syntax")
    return value
