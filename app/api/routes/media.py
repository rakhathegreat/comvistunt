"""Endpoints that handle image capture, landmarking, and streaming."""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.storage import (
    CAPTURE_IMAGE_PATH,
    LANDMARK_DIR,
)
from camera import (
    capture,
    generate_frames,
    get_latest_measurement,
    enable_stream,
    disable_stream,
)
from model.comvistunt import (
    draw_landmarks,
    get_landmarks,
)

router = APIRouter(tags=["Media"])
STREAM_INTERVAL_SECONDS = 0.5  # 2x per detik


def image_to_base64(image_path: str) -> str:
    """Konversi gambar lokal ke base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@router.post("/capture")
async def capture_image():
    """Capture image and return the frame as base64."""
    try:
        capture()

        if not CAPTURE_IMAGE_PATH.exists():
            raise FileNotFoundError("Captured frame was not saved.")

        raw_image_base64 = image_to_base64(str(CAPTURE_IMAGE_PATH))
        annotated_image_base64 = None

        landmarks = get_landmarks(str(CAPTURE_IMAGE_PATH))
        if landmarks:
            result = draw_landmarks(str(CAPTURE_IMAGE_PATH), landmarks, LANDMARK_DIR)
            annotated_path = LANDMARK_DIR / result if result else None
            if annotated_path and annotated_path.exists():
                annotated_image_base64 = image_to_base64(str(annotated_path))

        return {
            "image": annotated_image_base64 or raw_image_base64,
        }
    except Exception as exc:  # pragma: no cover - defensive coding
        return JSONResponse(content={"message": str(exc)}, status_code=500)


@router.get("/video")
async def video_stream(request: Request) -> StreamingResponse:
    """Live preview: MJPEG stream dengan overlay tinggi.

    - Akan berhenti jika:
      * client disconnect (request.is_disconnected())
      * atau endpoint /video/stop dipanggil (disable_stream()).
    """

    # setiap kali ada client baru, izinkan stream lagi
    enable_stream()

    async def frame_streamer():
        gen = generate_frames()  # generator sinkron dari camera.py
        try:
            for frame in gen:
                # cek apakah client sudah disconnect
                if await request.is_disconnected():
                    print("[Video] Client disconnected, stop streaming")
                    break

                # kirim frame ke client
                yield frame

                # beri kesempatan ke event loop
                await asyncio.sleep(0)
        finally:
            # pastikan generator ditutup -> finally di generate_frames() terpanggil
            if hasattr(gen, "close"):
                gen.close()
                print("[Video] Generator closed")

    return StreamingResponse(
        frame_streamer(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/video/stop")
async def video_stop():
    """Endpoint untuk mematikan stream video secara paksa dari luar.

    Misalnya dipanggil ketika user klik tombol "Putuskan kamera".
    """
    disable_stream()
    return {"status": "ok", "message": "Video stream will be stopped."}


@router.get("/video/metrics")
async def video_metrics():
    """Polling endpoint: ambil pengukuran terakhir sebagai JSON."""
    measurement = get_latest_measurement()
    payload = _build_metrics_payload(measurement)
    if payload["status"] == "no_data":
        return payload
    return payload


async def _metrics_event_stream(
    interval: float = STREAM_INTERVAL_SECONDS,
) -> AsyncGenerator[str, None]:
    """SSE: kirim data tinggi/berat realtime dalam format JSON string."""
    try:
        while True:
            payload = _build_metrics_payload(get_latest_measurement())
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print("[Metrics] Stream cancelled (client disconnected)")
        raise


@router.get("/video/metrics/stream")
async def video_metrics_stream():
    """Realtime metrics stream: konsumsi via EventSource di frontend."""
    return StreamingResponse(_metrics_event_stream(), media_type="text/event-stream")


def _build_metrics_payload(measurement: Dict[str, Any]):
    """Normalisasi payload pengukuran tinggi/berat."""
    height_cm = measurement.get("height_cm")
    updated_at = measurement.get("updated_at")
    has_landmarks = measurement.get("has_landmarks")

    if not updated_at:
        return {
            "status": "no_data",
            "message": "Belum ada data dari stream.",
            "height": None,
            "weight": None,
            "has_landmarks": False,
            "updated_at": None,
        }

    weight = get_weight(height_cm) if height_cm is not None else None
    return {
        "status": "success" if height_cm is not None else "no_landmark",
        "height": height_cm,
        "weight": weight,
        "has_landmarks": bool(has_landmarks),
        "updated_at": updated_at,
    }
