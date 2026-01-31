"""Endpoints that handle image capture, landmarking, and streaming."""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.schemas import StuntingRiskRequest, StuntingRiskResponse
from camera import (
    generate_frames,
    get_latest_measurement,
    enable_stream,
    disable_stream,
)
from model.comvistunt import get_haz, get_weight

router = APIRouter(tags=["Media"])
STREAM_INTERVAL_SECONDS = 0.5  # 2x per detik


def _extract_jpeg_from_mjpeg(chunk: bytes) -> bytes:
    """Strip MJPEG headers/boundaries and return raw JPEG bytes."""
    delimiter = b"\r\n\r\n"
    if delimiter not in chunk:
        raise ValueError("Invalid MJPEG frame: delimiter not found.")
    _, body = chunk.split(delimiter, 1)
    if body.endswith(b"\r\n"):
        body = body[:-2]
    return body


@router.post("/capture")
async def capture_image():
    """Grab a single frame from the video stream and return it as base64."""
    try:
        enable_stream()
        gen = generate_frames()
        try:
            chunk = next(gen)
        except StopIteration:
            raise RuntimeError("Video stream returned no frame.")
        finally:
            if hasattr(gen, "close"):
                gen.close()

        jpeg_bytes = _extract_jpeg_from_mjpeg(chunk)
        return {"image": base64.b64encode(jpeg_bytes).decode("utf-8")}
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


def _haz_lookup(value: float) -> Tuple[Optional[float], Optional[str]]:
    """Ambil HAZ (z-score) terdekat menggunakan tabel WHO; aman terhadap error."""
    try:
        return get_haz(value, gender="L", age=None)
    except Exception:
        return None, None


def _height_status_from_haz(
    haz_label: Optional[str], haz_score: Optional[float]
) -> str:
    """Terjemahkan HAZ ke status tinggi."""
    mapping = {
        "Severely stunted": "sangat pendek",
        "Stunted": "pendek",
        "Normal": "normal",
        "Tall": "tinggi",
    }
    if haz_label in mapping:
        return mapping[haz_label]

    if haz_score is None:
        return "normal"
    if haz_score < -3:
        return "sangat pendek"
    if haz_score < -2:
        return "pendek"
    if haz_score > 1:
        return "tinggi"
    return "normal"


def _weight_status_from_haz(haz_score: Optional[float]) -> str:
    """Kategorikan berat dengan ambang HAZ (tanpa BMI)."""
    if haz_score is None:
        return "normal"
    if haz_score < -3:
        return "sangat kurus"
    if haz_score < -2:
        return "kurus"
    if haz_score > 1:
        return "gemuk"
    return "normal"


@router.post("/stunting-risk", response_model=StuntingRiskResponse)
async def calculate_stunting_risk(height: float, weight: float):
    """Hitung risiko stunting hanya dari tinggi & berat badan berbasis HAZ."""
    height_haz, height_label = _haz_lookup(height)
    weight_haz, _ = _haz_lookup(weight)

    height_status = _height_status_from_haz(height_label, height_haz)
    weight_status = _weight_status_from_haz(weight_haz)

    stunting_risk = (
        "berisiko"
        if height_status in {"sangat pendek", "pendek"}
        or weight_status in {"sangat kurus", "kurus"}
        else "tidak berisiko"
    )

    return {
        "status": "success",
        "height_status": height_status,
        "weight_status": weight_status,
        "stunting_risk": stunting_risk,
        "height_haz": height_haz,
        "weight_haz": weight_haz,
    }
