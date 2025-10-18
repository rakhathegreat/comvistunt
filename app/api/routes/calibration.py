"""Endpoints related to camera calibration steps."""

from dataclasses import asdict

from fastapi import APIRouter, File, UploadFile

from app.api.schemas import CalibrationResponse, ErrorResponse
from app.services import calibrate_aruco_marker, calibrate_green_mat

router = APIRouter(
    prefix="/calibrate",
    tags=["Calibration"],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)


@router.post("/aruco", response_model=CalibrationResponse)
async def calibrate_aruco(image: UploadFile = File(...)) -> CalibrationResponse:
    """Calibrate camera using an ArUco marker."""

    outcome = await calibrate_aruco_marker(image)

    return CalibrationResponse(status="success", message="Calibration Success.", **asdict(outcome))


@router.post("/green_mat", response_model=CalibrationResponse)
async def calibrate_green_mat_endpoint(
    image: UploadFile = File(...),
) -> CalibrationResponse:
    """Calibrate camera using the green mat reference image."""

    outcome = await calibrate_green_mat(image)

    return CalibrationResponse(status="success", message="Calibration Success.", **asdict(outcome))
