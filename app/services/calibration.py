"""Services for camera calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import UploadFile

from calibration import aruco, green_mat
from config_manager import set_config

from app.core.storage import (
    ARUCO_IMAGE_PATH,
    GREEN_MAT_IMAGE_PATH,
    save_upload_file,
)

from .exceptions import NotFoundError


@dataclass(slots=True)
class CalibrationOutcome:
    """Structured data returned from calibration services."""

    result: float
    file_path: str


async def calibrate_aruco_marker(image: UploadFile) -> CalibrationOutcome:
    """Calibrate the camera using an uploaded ArUco marker image."""

    destination = ARUCO_IMAGE_PATH
    await save_upload_file(image, destination)

    output = aruco(str(destination))
    if not output or output[0] is None:
        raise NotFoundError("Calibration Failed. Marker not detected.")

    result, file_path = output
    set_config("CM_PER_PX", result)

    return CalibrationOutcome(result=result, file_path=file_path)


async def calibrate_green_mat(image: UploadFile) -> CalibrationOutcome:
    """Calibrate the camera using an uploaded green mat reference image."""

    destination = GREEN_MAT_IMAGE_PATH
    await save_upload_file(image, destination)

    result, file_path = green_mat(str(destination))
    if result is None:
        raise NotFoundError("Calibration Failed.")

    return CalibrationOutcome(result=result, file_path=file_path)


__all__ = ["CalibrationOutcome", "calibrate_aruco_marker", "calibrate_green_mat"]
