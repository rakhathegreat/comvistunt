"""Endpoints that handle image capture, landmarking, and streaming."""

import base64
import os

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from camera import generate_frames, capture
from model.comvistunt import draw_landmarks, get_landmarks
from app.core.storage import CAPTURE_IMAGE_PATH, save_upload_file, LANDMARK_DIR, RESULT_LANDMARK_PATH

router = APIRouter(tags=["Media"])

def image_to_base64(image_path: str) -> str:
    """Mengonversi gambar ke format base64."""
    with open(image_path, "rb") as image_file:
        # Membaca file gambar dalam mode biner
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


@router.post("/capture")
async def capture_image():
    """Capture image and return the annotated landmark result."""

    try:
        capture()
        landmarks = str(CAPTURE_IMAGE_PATH)
        result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks, LANDMARK_DIR)

        return {
            "status": "success",
            "message": "Image captured.",
            "image": result,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)

@router.post("/captureweb")
async def capture_image():
    """Persist an uploaded image and return the annotated landmark result."""

    try:
        capture()
        landmarks = get_landmarks("image/baby6.jpg")
        result = draw_landmarks("image/baby6.jpg", landmarks, LANDMARK_DIR)

        if os.path.exists(RESULT_LANDMARK_PATH):
        # Mengambil gambar dalam base64
            base64_image = image_to_base64(RESULT_LANDMARK_PATH)
        else:
            return JSONResponse(
                status_code=404,
                content={"status": "failed", "message": "Can't convert image to base64"},
            )
        return {
            "status": "success",
            "message": "Image captured.",
            "image": base64_image,
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
