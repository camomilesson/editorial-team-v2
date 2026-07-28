"""Sanitized errors for model-backed agent boundaries."""

from editorial_team.errors import ServiceError


class AgentError(ServiceError):
    """A model-backed agent could not produce a valid result."""
