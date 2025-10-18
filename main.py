"""FastAPI application entrypoint with structured routing."""

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import (
    CSPMiddleware,
    LANDMARK_DIR,
    FastAPI,
    analysis_router,
    calibration_router,
    media_router,
    system_router,
)


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    application = FastAPI()
    application.add_middleware(CSPMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.mount(
        "/static", StaticFiles(directory=str(LANDMARK_DIR)), name="static"
    )

    application.include_router(system_router)
    application.include_router(calibration_router)
    application.include_router(media_router)
    application.include_router(analysis_router)

    return application


app = create_app()

__all__ = ["app", "create_app"]
