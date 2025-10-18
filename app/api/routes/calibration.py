"""Endpoints related to camera calibration steps."""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from calibration import aruco, green_mat
from config_manager import set_config
from app.core.storage import ARUCO_IMAGE_PATH, GREEN_MAT_IMAGE_PATH, save_upload_file

router = APIRouter(prefix="/calibrate", tags=["Calibration"])


@router.post("/aruco")
async def calibrate_aruco(image: UploadFile = File(...)):
    """Calibrate camera using an ArUco marker."""

    try:
        destination = ARUCO_IMAGE_PATH
        await save_upload_file(image, destination)

        output = aruco(str(destination))
        if not output or output[0] is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "failed",
                    "message": "Calibration Failed. Marker not detected.",
                },
            )

        result, file_path = output
        set_config("CM_PER_PX", result)

        return {
            "status": "success",
            "message": "Calibration Success.",
            "result": result,
            "file_path": file_path,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)


@router.post("/green_mat")
async def calibrate_green_mat(image: UploadFile = File(...)):
    """Calibrate camera using the green mat reference image."""

    try:
        destination = GREEN_MAT_IMAGE_PATH
        await save_upload_file(image, destination)

        result, file_path = green_mat(str(destination))

        if result is None:
            return JSONResponse(
                status_code=404,
                content={"status": "failed", "message": "Calibration Failed."},
            )

        return {
            "status": "success",
            "message": "Calibration Success.",
            "result": result,
            "file_path": file_path,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)
