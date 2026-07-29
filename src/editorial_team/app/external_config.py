"""Sanitized environment configuration for the external brief server."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EXTERNAL_API_HOST = "127.0.0.1"
DEFAULT_EXTERNAL_API_PORT = 8080


class ExternalApiConfigurationError(RuntimeError):
    """External API configuration is absent or invalid."""


@dataclass(frozen=True)
class ExternalApiConfiguration:
    """Validated external API settings."""

    token: str
    host: str = DEFAULT_EXTERNAL_API_HOST
    port: int = DEFAULT_EXTERNAL_API_PORT


def load_external_api_configuration() -> ExternalApiConfiguration:
    """Load external API settings without exposing their values."""

    token = os.getenv("EDITORIAL_EXTERNAL_API_TOKEN", "")
    if not isinstance(token, str) or not token.strip():
        raise ExternalApiConfigurationError("Required external API configuration is missing")
    host = os.getenv("EDITORIAL_EXTERNAL_API_HOST", DEFAULT_EXTERNAL_API_HOST)
    if not isinstance(host, str) or not host.strip():
        raise ExternalApiConfigurationError("External API host is invalid")
    raw_port = os.getenv("EDITORIAL_EXTERNAL_API_PORT", str(DEFAULT_EXTERNAL_API_PORT))
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        raise ExternalApiConfigurationError("External API port is invalid") from None
    if isinstance(raw_port, bool) or not 1 <= port <= 65535 or str(port) != raw_port.strip():
        raise ExternalApiConfigurationError("External API port is invalid")
    return ExternalApiConfiguration(token=token, host=host.strip(), port=port)
