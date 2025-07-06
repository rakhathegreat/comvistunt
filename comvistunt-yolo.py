import cv2, math, time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# CONFIG
POSE_LANDMARK_VIS_THRESH = 0.999

# DATA CLASSES
@dataclass
class Landmark:
    head: Tuple[float, float];
    shoulders: Tuple[float, float];
    hips: Tuple[float, float];
    ankle: Tuple[float, float];

# FUNCTION
def get_landmarks(image) ->Optional[Landmark]:
    try: # coba import mediapipe
        import mediapipe as mp 
    except ImportError: # kalau tidak bisa tampilkan error
        print("Modul Mediapipe tidak ditemukan silahkan install terlebih dahulu")
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # convert gambar ke RGB

        if not res.pose_landmarks: # kalau tidak ada landmark return none
            return None
        
        lm = res.pose_landmarks.landmark # ambil landmark

        if lm[mp.solutions.pose.PoseLandmark.NOSE].visibility < POSE_LANDMARK_VIS_THRESH: # jika visibility landmark hidung kurang dari 0.999 maka return none
            return None
        
        h,w,_ = image.shape # ambil ukuran dan channel warna

        def to_xy(idx): L=lm[idx]; return int(L.x*w), int(L.y*h) # konversi landmark ke format xy

        return Landmark( # return landmark head, shoulders, hips, ankle
            head=to_xy(mp.solutions.pose.PoseLandmark.NOSE),
            shoulders=to_xy(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
            hips=to_xy(mp.solutions.pose.PoseLandmark.LEFT_HIP),
            ankle=to_xy(mp.solutions.pose.PoseLandmark.LEFT_ANKLE)
        )

def get_height(lms: Landmark, cm_per_px: float) -> float:
    return cm_per_px * math.hypot(lms.head[0]-lms.ankle[0], lms.head[1]-lms.ankle[1])


        




