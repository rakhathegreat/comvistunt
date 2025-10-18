"""Filesystem utilities shared across API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import UploadFile


UPLOAD_DIR = Path("capture")
LANDMARK_DIR = UPLOAD_DIR / "landmark"
CALIBRATION_DIR = UPLOAD_DIR / "calibration"
CAPTURE_IMAGE_PATH = UPLOAD_DIR / "captured.png"
RESULT_LANDMARK_PATH = LANDMARK_DIR / "draw-landmark.png"
ARUCO_IMAGE_PATH = CALIBRATION_DIR / "aruco.png"
GREEN_MAT_IMAGE_PATH = CALIBRATION_DIR / "green_mat.png"


def _ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories(
    (
        UPLOAD_DIR,
        LANDMARK_DIR,
        CALIBRATION_DIR,
    )
)


async def save_upload_file(upload: UploadFile, destination: Path) -> Path:
    """Persist an :class:`~fastapi.UploadFile` to ``destination`` asynchronously."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = await upload.read()
    with destination.open("wb") as buffer:
        buffer.write(contents)
    return destination
