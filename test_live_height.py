"""Manual webcam test for live height estimation.

Run this script directly to open the webcam, draw pose landmarks, and display
the estimated height overlay. Press 'q' to quit.
"""

import cv2

from config_manager import get_config
from model.comvistunt import (
    draw_landmarks_on_frame,
    get_height,
    get_landmarks_from_frame,
)


def main():
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        print("mediapipe belum terpasang. Install mediapipe terlebih dahulu.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam.")
        return

    cm_per_px = float(get_config("CM_PER_PX") or 0)
    pose = mp.solutions.pose.Pose(static_image_mode=False)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca frame dari webcam.")
                break

            lms = get_landmarks_from_frame(frame, pose)
            height_cm = (
                get_height(lms, cm_per_px) if lms and cm_per_px > 0 else None
            )

            if lms:
                frame = draw_landmarks_on_frame(frame, lms)

            if height_cm is not None:
                cv2.putText(
                    frame,
                    f"Tinggi: {height_cm:.2f} cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Live Height (Tekan 'q' untuk keluar)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
