"""Pydantic response models shared across API routes."""

from typing import Literal

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    status: Literal["success"]
    message: str


class CalibrationResponse(StatusResponse):
    result: float
    file_path: str


class CaptureResponse(StatusResponse):
    image: str


class AnalysisResponse(StatusResponse):
    height: float
    haz: float
    weight: float


class ErrorResponse(BaseModel):
    status: Literal["failed"]
    message: str


__all__ = [
    "AnalysisResponse",
    "CalibrationResponse",
    "CaptureResponse",
    "ErrorResponse",
    "MessageResponse",
    "StatusResponse",
]
