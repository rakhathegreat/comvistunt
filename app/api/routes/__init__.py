"""Aggregate API router that composes all endpoint modules."""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .calibration import router as calibration_router
from .media import router as media_router
from .system import router as system_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(calibration_router)
api_router.include_router(media_router)
api_router.include_router(analysis_router)

__all__ = ["api_router"]
