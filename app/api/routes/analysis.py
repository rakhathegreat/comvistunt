"""Endpoints performing growth and health analysis."""

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from config_manager import get_config
from model.comvistunt import get_haz, get_height, get_landmarks, get_weight
from app.core.storage import CAPTURE_IMAGE_PATH

router = APIRouter(tags=["Analysis"])


@router.post("/analyze")
async def analyze(gender: str = Form(...), age: int = Form(...)):
    """Calculate height, HAZ, and weight from the captured image."""

    try:
        file_path = CAPTURE_IMAGE_PATH
        lms = get_landmarks(str(file_path))
        height = get_height(lms, get_config("CM_PER_PX"))
        z_score, _ = get_haz(height, gender, age)
        weight = get_weight(height)

        if height is None or z_score is None or weight is None:
            return JSONResponse(
                status_code=404,
                content={"status": "failed", "message": "Can't analyze."},
            )

        return {
            "status": "success",
            "message": "Landmark Obtained.",
            "height": height,
            "haz": z_score,
            "weight": weight,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(
            content={"status": "failed", "message": str(exc)}, status_code=500
        )
