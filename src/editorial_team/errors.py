"""Sanitized errors for service boundaries."""


class ServiceError(RuntimeError):
    """Base error safe to show at a service boundary."""


class EntityNotFoundError(ServiceError):
    """A requested entity does not exist."""


class DuplicateEntityError(ServiceError):
    """A unique entity already exists."""


class AuthorizationError(ServiceError):
    """The caller is not permitted to perform an operation."""


class DataValidationError(ServiceError):
    """Structured data cannot be safely validated."""
