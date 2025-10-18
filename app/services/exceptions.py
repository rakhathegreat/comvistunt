"""Custom exceptions raised by service layer functions."""

from __future__ import annotations


class ServiceError(RuntimeError):
    """Base class for recoverable service errors."""


class NotFoundError(ServiceError):
    """Raised when a requested resource or computation is unavailable."""
