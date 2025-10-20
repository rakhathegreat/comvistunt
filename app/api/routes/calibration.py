"""Endpoints related to camera calibration steps."""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from camera import capture

from calibration import aruco, green_mat
from config_manager import set_config
from app.core.storage import CAPTURE_IMAGE_PATH, GREEN_MAT_IMAGE_PATH, save_upload_file

router = APIRouter(prefix="/calibrate", tags=["Calibration"])


@router.post("/aruco")
async def calibrate_aruco(file: UploadFile = File(...)):
    """
    Terima 1 file gambar (jpeg/png) via multipart/form-data
    """
    try:
        file_path = await save_upload_file(file, CAPTURE_IMAGE_PATH)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

    output = aruco(file_path)
    if not output or output[0] is None:
        return JSONResponse(
            status_code=404,
            content={"status": "failed", "message": "Marker not detected"},
        )

    result, _ = output
    set_config("CM_PER_PX", result)
    return {"status": "success", "result": result}

