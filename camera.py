import cv2
import os
import time

from config_manager import get_config
from model.comvistunt import get_height, get_landmarks_from_frame

_latest_measurement = {"height_cm": None, "has_landmarks": False, "updated_at": None}
_console_log_state = {"last_height": None, "last_print": 0.0}

# Flag global untuk mengontrol apakah stream masih boleh jalan
_stream_enabled = True


def enable_stream():
    """Izinkan stream berjalan (dipanggil sebelum mulai /video)."""
    global _stream_enabled
    _stream_enabled = True


def disable_stream():
    """Minta stream berhenti (dicek di generate_frames)."""
    global _stream_enabled
    _stream_enabled = False


def is_stream_enabled() -> bool:
    """Cek status stream global."""
    return _stream_enabled


def get_latest_measurement():
    """Return the last computed height (if any) from the video stream."""
    return _latest_measurement


def generate_frames():
    """Stream frames while computing live height estimation (tanpa overlay).

    Loop akan berhenti jika:
    - kamera gagal baca frame, atau
    - generator di-close (GeneratorExit), atau
    - flag global _stream_enabled di-set False lewat /video/stop.
    """

    try:
        import mediapipe as mp  # noqa: F401 - used to initialize pose below
    except ImportError:
        mp = None

    camera = cv2.VideoCapture(0)
    pose = mp.solutions.pose.Pose(static_image_mode=False) if mp else None
    cm_per_px = float(get_config("CM_PER_PX") or 0)

    try:
        while True:
            # kalau stream dimatikan lewat endpoint /video/stop
            if not is_stream_enabled():
                print("[Video] Stream disabled by endpoint, stop loop")
                break

            success, frame = camera.read()
            if not success:
                break

            height_cm = None
            if pose:
                lms = get_landmarks_from_frame(frame, pose)
                if lms:
                    height_cm = get_height(lms, cm_per_px) if cm_per_px > 0 else None

            now = time.time()
            _latest_measurement.update(
                {
                    "height_cm": height_cm,
                    "has_landmarks": bool(height_cm is not None),
                    "updated_at": now,
                }
            )

            # Cetak ke console secara terkontrol
            should_log = False
            if height_cm is not None:
                last_height = _console_log_state["last_height"]
                delta_height = (
                    abs(height_cm - last_height) if last_height is not None else None
                )
                if (
                    last_height is None
                    or (delta_height is not None and delta_height >= 0.5)
                    or (now - _console_log_state["last_print"] > 2)
                ):
                    should_log = True
                    _console_log_state["last_height"] = height_cm
            else:
                if now - _console_log_state["last_print"] > 3:
                    should_log = True

            if should_log:
                msg = (
                    f"[Live] Tinggi: {height_cm:.2f} cm"
                    if height_cm is not None
                    else "[Live] Landmark belum terdeteksi"
                )
                print(msg)
                _console_log_state["last_print"] = now

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    except GeneratorExit:
        # Dipanggil ketika gen.close() dipanggil dari endpoint
        print("[Video] GeneratorExit: stop streaming")

    finally:
        if pose:
            pose.close()
        camera.release()
        print("[Video] Camera released")


def capture():
    # Membuka kamera (0 = kamera default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Gagal mengakses kamera!")
        return

    # Membaca satu frame dari kamera
    ret, frame = cap.read()

    if ret:
        os.makedirs("capture", exist_ok=True)
        # Menyimpan gambar ke file
        cv2.imwrite("capture/captured.png", frame)
        print("Gambar berhasil disimpan!")
    else:
        print("Gagal mengambil gambar!")

    # Melepaskan kamera
    cap.release()
