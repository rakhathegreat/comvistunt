"""Pydantic response models shared across API routes."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


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


class StuntingRiskRequest(BaseModel):
    height_cm: float = Field(..., gt=0, description="Tinggi badan dalam centimeter.")
    weight_kg: float = Field(..., gt=0, description="Berat badan dalam kilogram.")


class StuntingRiskResponse(BaseModel):
    status: Literal["success"]
    height_status: Literal["sangat pendek", "pendek", "normal", "tinggi"]
    weight_status: Literal["sangat kurus", "kurus", "normal", "gemuk"]
    stunting_risk: Literal["berisiko", "tidak berisiko"]
    height_haz: Optional[float] = Field(None, description="Z-score tinggi.")
    weight_haz: Optional[float] = Field(None, description="Z-score berat (berbasis HAZ lookup).")


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
    "StuntingRiskRequest",
    "StuntingRiskResponse",
]
