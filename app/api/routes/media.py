"""Endpoints that handle image capture, landmarking, and streaming."""

from dataclasses import asdict

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import CaptureResponse, ErrorResponse
from app.services import generate_frames, generate_landmark_image, save_capture

router = APIRouter(
    tags=["Media"],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)


@router.post("/capture", response_model=CaptureResponse)
async def capture_image(image: UploadFile = File(...)) -> CaptureResponse:
    """Persist an uploaded image and return the annotated landmark result."""

    outcome = await save_capture(image)

    return CaptureResponse(status="success", message="Image Uploaded.", **asdict(outcome))


@router.post("/get_landmark", response_model=CaptureResponse)
async def get_landmark() -> CaptureResponse:
    """Retrieve landmarks for the latest captured image."""

    outcome = generate_landmark_image()

    return CaptureResponse(status="success", message="Landmark Obtained.", **asdict(outcome))


@router.get("/video")
async def video_stream() -> StreamingResponse:
    """Stream MJPEG frames from the configured camera source."""

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
