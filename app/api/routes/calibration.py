"""Endpoints related to camera calibration steps."""

from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import CalibrationResponse, ErrorResponse
from app.services import (
    NotFoundError,
    ServiceError,
    calibrate_aruco_marker,
    calibrate_green_mat,
)

router = APIRouter(
    prefix="/calibrate",
    tags=["Calibration"],
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@router.post("/aruco", response_model=CalibrationResponse)
async def calibrate_aruco(image: UploadFile = File(...)) -> CalibrationResponse:
    """Calibrate camera using an ArUco marker."""

    try:
        outcome = await calibrate_aruco_marker(image)
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

    return CalibrationResponse(status="success", message="Calibration Success.", **asdict(outcome))


@router.post("/green_mat", response_model=CalibrationResponse)
async def calibrate_green_mat_endpoint(
    image: UploadFile = File(...),
) -> CalibrationResponse:
    """Calibrate camera using the green mat reference image."""

    try:
        outcome = await calibrate_green_mat(image)
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

    return CalibrationResponse(status="success", message="Calibration Success.", **asdict(outcome))
