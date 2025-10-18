"""Application package grouping API routes and shared utilities."""

from fastapi import FastAPI

from .api.routes.analysis import router as analysis_router
from .api.routes.calibration import router as calibration_router
from .api.routes.media import router as media_router
from .api.routes.system import router as system_router
from .core.middleware import CSPMiddleware
from .core.storage import LANDMARK_DIR

__all__ = [
    "FastAPI",
    "analysis_router",
    "calibration_router",
    "media_router",
    "system_router",
    "CSPMiddleware",
    "LANDMARK_DIR",
]
