import cv2, math, time, os, datetime, glob
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from config_manager import get_config


# DATA CLASSES
@dataclass
class Landmark:
    head: Tuple[float, float]
    nose: Tuple[float, float]
    shoulder: Tuple[float, float]
    hip: Tuple[float, float]
    right_hip: Tuple[float, float]
    knee: Tuple[float, float]
    ankle: Tuple[float, float]
    heel: Tuple[float, float]

# HEIGHT ESTIMATION
def _pose_landmarks_to_struct(mp_pose, lm, shape) -> Optional[Landmark]:
    """Convert MediaPipe landmarks into our Landmark dataclass."""

    def landmark_to_px(landmark, current_shape):
        return (int(landmark.x * current_shape[1]), int(landmark.y * current_shape[0]))

    nose = landmark_to_px(lm[mp_pose.PoseLandmark.NOSE], shape)
    left_eye = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_EYE], shape)
    right_eye = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_EYE], shape)
    left_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], shape)
    right_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], shape)
    left_hip = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_HIP], shape)
    right_hip = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HIP], shape)
    knee = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE], shape)
    ankle = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE], shape)
    heel = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HEEL], shape)

    shoulder_px = tuple(np.mean([left_shoulder, right_shoulder], axis=0).astype(int))
    hip_px = tuple(np.mean([left_hip, right_hip], axis=0).astype(int))
    eyes_px = tuple(np.mean([left_eye, right_eye], axis=0).astype(int))

    vec_direction = np.array(eyes_px) - np.array(nose)
    vec_norm = np.linalg.norm(vec_direction)
    if vec_norm == 0:
        return None
    head_peak = tuple(
        (
            np.array(nose)
            + (vec_direction / vec_norm) * abs(shoulder_px[1] - hip_px[1]) / 2
        ).astype(int)
    )

    return Landmark(
        head=head_peak,
        nose=nose,
        shoulder=shoulder_px,
        hip=hip_px,
        right_hip=right_hip,
        knee=knee,
        ankle=ankle,
        heel=heel,
    )


def get_landmarks(image: str) -> Optional[Landmark]:
    try:
        import mediapipe as mp
    except ImportError:
        return None

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        img = cv2.imread(image)
        if img is None:
            return None

        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            return _pose_landmarks_to_struct(mp_pose, lm, img.shape)
    return None


def get_landmarks_from_frame(frame, pose=None) -> Optional[Landmark]:
    """
    Real-time version of get_landmarks that processes an in-memory frame.
    Pass an existing MediaPipe pose instance to reuse it across frames.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    mp_pose = mp.solutions.pose
    should_close_pose = False
    if pose is None:
        pose = mp_pose.Pose(static_image_mode=False)
        should_close_pose = True

    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if should_close_pose:
        pose.close()

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        return _pose_landmarks_to_struct(mp_pose, lm, frame.shape)
    return None

def _draw_landmark_lines(img, lms: Landmark):
    """Draw measurement guide lines on top of a frame."""

    cv2.line(img, lms.nose, lms.head, (0, 0, 255), 2)
    cv2.line(img, lms.nose, lms.hip, (0, 0, 255), 2)
    cv2.line(img, lms.right_hip, lms.knee, (0, 0, 255), 2)
    cv2.line(img, lms.knee, lms.ankle, (0, 0, 255), 2)
    cv2.line(img, lms.ankle, lms.heel, (0, 0, 255), 2)
    return img


def draw_landmarks_on_frame(frame, lms: Landmark):
    """Draw landmarks on an in-memory frame for real-time display."""

    if lms is None:
        return frame

    if None in [
        lms.nose,
        lms.head,
        lms.hip,
        lms.right_hip,
        lms.knee,
        lms.ankle,
        lms.heel,
    ]:
        return frame

    return _draw_landmark_lines(frame, lms)


def draw_landmarks(image: str, lms: 'Landmark', dir: str) -> str:
    try:
        # Baca gambar
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan atau tidak bisa dibaca: {image}")

        # Pastikan semua landmark tidak None
        if None in [lms.nose, lms.head, lms.hip, lms.right_hip, lms.knee, lms.ankle, lms.heel]:
            raise ValueError("Beberapa titik landmark belum diatur (None)")

        # Gambar garis
        _draw_landmark_lines(img, lms)

        output_dir = dir

        # Simpan hasil
        file = f"draw-landmark.png"
        success = cv2.imwrite(os.path.join(output_dir, file), img)

        if not success:
            raise IOError(f"Gagal menyimpan gambar ke: {output_dir}")

        return file

    except Exception as e:
        print(f"[ERROR] draw_landmark: {e}")
        return None
    
def get_height(lms: Landmark, ref: float) -> float:
    
    def pixel_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    
    d1 = pixel_distance(lms.nose, lms.head)
    d2 = pixel_distance(lms.nose, lms.hip)
    d3 = pixel_distance(lms.right_hip, lms.knee)
    d4 = pixel_distance(lms.knee, lms.ankle)
    d5 = pixel_distance(lms.ankle, lms.heel)

    return ref * (d1 + d2 + d3 + d4 + d5) 

def get_weight(height:float) -> float:
    bmi = 22
    return bmi * (height / 100)**2

# HAZ ESTIMATION
def get_haz(height: float, gender: str = "L", age: Optional[int] = None) -> Tuple[float, str]:
    """Calculate height-for-age Z-score (HAZ) and label."""
    gender_norm = gender.lower()
    if gender_norm in {"l", "male", "m"}:
        who_df = pd.read_csv("haz/HAZ_TABLE_BOYS.csv")
    else:
        who_df = pd.read_csv("haz/HAZ_TABLE_GIRLS.csv")

    if age is not None:
        row = who_df.loc[who_df["age"] == age]
        if row.empty:
            # Fallback: gunakan umur terdekat jika umur tidak ada di tabel
            nearest_idx = (who_df["age"] - age).abs().idxmin()
            row = who_df.loc[[nearest_idx]]
    else:
        # Jika umur tidak diberikan, pakai baris dengan median paling dekat ke input
        nearest_idx = (who_df["median"] - height).abs().idxmin()
        row = who_df.loc[[nearest_idx]]

    median = row["median"].values[0]
    sd = row["sd"].values[0]

    z = (height - median) / sd

    if z < -3:
        label = "Severely stunted"
    elif z < -2:
        label = "Stunted"
    elif z > 1:
        label = "Tall"
    else:
        label = "Normal"

    return (z, label)
