"""Application factory for constructing the FastAPI instance."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import api_router
from .core.errors import register_exception_handlers
from .core.middleware import CSPMiddleware
from .core.storage import LANDMARK_DIR


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app = FastAPI()

    app.add_middleware(CSPMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(LANDMARK_DIR)), name="static")

    register_exception_handlers(app)
    app.include_router(api_router)

    return app


__all__ = ["create_app"]
