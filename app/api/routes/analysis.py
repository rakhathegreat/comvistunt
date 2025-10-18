"""Endpoints performing growth and health analysis."""

from dataclasses import asdict

from fastapi import APIRouter, Form, HTTPException

from app.api.schemas import AnalysisResponse, ErrorResponse
from app.services import NotFoundError, ServiceError, analyze_growth

router = APIRouter(
    tags=["Analysis"],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(gender: str = Form(...), age: int = Form(...)) -> AnalysisResponse:
    """Calculate height, HAZ, and weight from the captured image."""

    try:
        outcome = analyze_growth(gender, age)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"status": "failed", "message": str(exc)},
        ) from exc
    except ServiceError as exc:
        raise HTTPException(
            status_code=400,
            detail={"status": "failed", "message": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive coding
        raise HTTPException(
            status_code=500,
            detail={"status": "failed", "message": str(exc)},
        ) from exc

    return AnalysisResponse(
        status="success",
        message="Landmark Obtained.",
        **asdict(outcome),
    )
