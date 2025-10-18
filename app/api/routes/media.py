"""Endpoints that handle image capture, landmarking, and streaming."""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from camera import generate_frames
from model.comvistunt import draw_landmarks, get_landmarks
from app.core.storage import CAPTURE_IMAGE_PATH, save_upload_file

router = APIRouter(tags=["Media"])


@router.post("/capture")
async def capture_image(image: UploadFile = File(...)):
    """Persist an uploaded image and return the annotated landmark result."""

    try:
        await save_upload_file(image, CAPTURE_IMAGE_PATH)
        landmarks = get_landmarks(str(CAPTURE_IMAGE_PATH))
        result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks)

        return {
            "status": "success",
            "message": "Image Uploaded.",
            "image": result,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)


@router.post("/get_landmark")
async def get_landmark():
    """Retrieve landmarks for the latest captured image."""

    try:
        landmarks = get_landmarks(str(CAPTURE_IMAGE_PATH))
        result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks)

        if result is None:
            return JSONResponse(
                status_code=404,
                content={"status": "failed", "message": "Can't get landmark."},
            )

        return {
            "status": "success",
            "message": "Landmark Obtained.",
            "image": result,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)


@router.get("/video")
async def video_stream() -> StreamingResponse:
    """Stream MJPEG frames from the configured camera source."""

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
