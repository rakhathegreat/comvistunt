"""Services that orchestrate media capture and landmark operations."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import UploadFile

from camera import generate_frames
from model.comvistunt import draw_landmarks, get_landmarks

from app.core.storage import CAPTURE_IMAGE_PATH, save_upload_file

from .exceptions import NotFoundError


@dataclass(slots=True)
class CaptureOutcome:
    """Represents the result of persisting an uploaded capture image."""

    image: str


async def save_capture(image: UploadFile) -> CaptureOutcome:
    """Persist the uploaded image and generate the landmark visualization."""

    await save_upload_file(image, CAPTURE_IMAGE_PATH)
    landmarks = get_landmarks(str(CAPTURE_IMAGE_PATH))
    result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks)

    if result is None:
        raise NotFoundError("Image upload failed.")

    return CaptureOutcome(image=result)


@dataclass(slots=True)
class LandmarkOutcome:
    """Represents the rendered landmark result for the stored capture."""

    image: str


def generate_landmark_image() -> LandmarkOutcome:
    """Render the latest landmark overlay for the stored capture image."""

    landmarks = get_landmarks(str(CAPTURE_IMAGE_PATH))
    result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks)

    if result is None:
        raise NotFoundError("Can't get landmark.")

    return LandmarkOutcome(image=result)


__all__ = ["CaptureOutcome", "LandmarkOutcome", "generate_landmark_image", "save_capture", "generate_frames"]
