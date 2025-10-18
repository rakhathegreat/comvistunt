"""System level endpoints such as health checks."""

from fastapi import APIRouter

from app.api.schemas import MessageResponse

router = APIRouter(tags=["System"])


@router.get("/", response_model=MessageResponse)
async def root() -> MessageResponse:
    """Basic hello world endpoint for smoke testing."""

    return MessageResponse(message="Hello World")


@router.get("/ping", response_model=MessageResponse)
async def ping() -> MessageResponse:
    """Return a pong response for uptime monitoring."""

    return MessageResponse(message="pong")
