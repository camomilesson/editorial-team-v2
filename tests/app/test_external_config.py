from __future__ import annotations

import pytest

from editorial_team.app.external_config import (
    DEFAULT_EXTERNAL_API_HOST,
    DEFAULT_EXTERNAL_API_PORT,
    ExternalApiConfigurationError,
    load_external_api_configuration,
)


def test_external_configuration_requires_token_and_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITORIAL_EXTERNAL_API_TOKEN", raising=False)

    with pytest.raises(
        ExternalApiConfigurationError,
        match=r"^Required external API configuration is missing$",
    ):
        load_external_api_configuration()

    monkeypatch.setenv("EDITORIAL_EXTERNAL_API_TOKEN", "placeholder-token")
    configuration = load_external_api_configuration()

    assert configuration.token == "placeholder-token"
    assert configuration.host == DEFAULT_EXTERNAL_API_HOST
    assert configuration.port == DEFAULT_EXTERNAL_API_PORT


@pytest.mark.parametrize("port", ["0", "65536", "8.5", "08080", "private-value"])
def test_external_configuration_rejects_invalid_port_safely(
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    monkeypatch.setenv("EDITORIAL_EXTERNAL_API_TOKEN", "placeholder-token")
    monkeypatch.setenv("EDITORIAL_EXTERNAL_API_PORT", port)

    with pytest.raises(ExternalApiConfigurationError) as caught:
        load_external_api_configuration()

    assert str(caught.value) == "External API port is invalid"
    assert port not in str(caught.value)
