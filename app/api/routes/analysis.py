"""Endpoints performing growth and health analysis."""

from dataclasses import asdict

from fastapi import APIRouter, Form

from app.api.schemas import AnalysisResponse, ErrorResponse
from app.services import analyze_growth

router = APIRouter(
    tags=["Analysis"],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(gender: str = Form(...), age: int = Form(...)) -> AnalysisResponse:
    """Calculate height, HAZ, and weight from the captured image."""

    outcome = analyze_growth(gender, age)

    return AnalysisResponse(
        status="success",
        message="Analysis completed.",
        **asdict(outcome),
    )
