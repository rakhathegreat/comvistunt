import cv2, math, time
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from config_manager import get_config


# DATA CLASSES
@dataclass
class Landmark:
    head: Tuple[float, float]
    shoulder: Tuple[float, float]
    hip: Tuple[float, float]
    right_hip: Tuple[float, float]
    knee: Tuple[float, float]
    ankle: Tuple[float, float]
    heel: Tuple[float, float]

# HEIGHT ESTIMATION
def get_landmark(image: str) -> Optional[Landmark]:
    try:
        import mediapipe as mp
    except ImportError:
        return None
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:  
        def landmark_to_px(landmark, shape):
            return (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
        

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Deteksi pose
        res = pose.process(img)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            shape = img.shape

            # Landmark penting
            left_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], shape)
            right_shoulder = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], shape)
            left_hip = landmark_to_px(lm[mp_pose.PoseLandmark.LEFT_HIP], shape)
            right_hip = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HIP], shape)
            knee = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE], shape)
            ankle = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE], shape)
            heel = landmark_to_px(lm[mp_pose.PoseLandmark.RIGHT_HEEL], shape)

            shoulder_px = tuple(np.mean([left_shoulder, right_shoulder], axis=0).astype(int))
            hip_px = tuple(np.mean([left_hip, right_hip], axis=0).astype(int))

            vec = np.array(shoulder_px) - np.array(hip_px)
            head_peak = tuple((np.array(hip_px) + vec * 1.83).astype(int))

            return Landmark(
                head=head_peak,
                shoulder=shoulder_px,
                hip=hip_px,
                right_hip=right_hip,
                knee=knee,
                ankle=ankle,
                heel=heel
            )

def get_height(lms: Landmark, ref: float) -> float:
    
    def pixel_distance(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))
    
    
    h1 = pixel_distance(lms.hip, lms.head)
    h2 = pixel_distance(lms.right_hip, lms.knee)
    h3 = pixel_distance(lms.knee, lms.ankle)
    h4 = pixel_distance(lms.ankle, lms.heel)

    return ref * (h1 + h2 + h3 + h4) 

        
# HAZ ESTIMATION
def get_haz(height: float, gender: str, age: int) -> Tuple[float, str]:
    if gender.lower() == "male":
        who_df = pd.read_csv("haz/HAZ_TABLE_BOYS.csv")
    else:
        who_df = pd.read_csv("haz/HAZ_TABLE_GIRLS.csv")

    median = who_df.loc[who_df["age"] == age, "median"].values[0]
    sd = who_df.loc[who_df["age"] == age, "sd"].values[0]

    z = (height - median) / sd

    if z < -3:
        label = "Severely stunted"
    elif z < -2:
        label = "Stunted"
    elif z > 1:
        label = "Tall"
    else:
        label = "Normal"

    return z, label



