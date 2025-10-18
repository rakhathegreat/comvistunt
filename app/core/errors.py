"""Exception handlers shared across the FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.exceptions import NotFoundError, ServiceError


def _format_error(exc: Exception) -> dict[str, str]:
    """Return a JSON-serialisable error payload."""

    return {"status": "failed", "message": str(exc)}


async def _handle_not_found(_: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content=_format_error(exc))


async def _handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=400, content=_format_error(exc))


def register_exception_handlers(app: FastAPI) -> None:
    """Attach shared exception handlers to ``app``."""

    app.add_exception_handler(NotFoundError, _handle_not_found)
    app.add_exception_handler(ServiceError, _handle_service_error)


__all__ = ["register_exception_handlers"]
