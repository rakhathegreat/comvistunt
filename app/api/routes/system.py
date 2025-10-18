"""System level endpoints such as health checks."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["System"])


@router.get("/")
async def root():
    """Basic hello world endpoint for smoke testing."""

    return {"message": "Hello World", "status": 200}


@router.get("/ping")
async def ping() -> JSONResponse:
    """Return a pong response for uptime monitoring."""

    return JSONResponse(content={"message": "pong"}, status_code=200)
