# comvistunt_core.py – Dual-mode, rule-based anthropometry core
import os, cv2, json, math, time
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# ---------- CONFIG ----------------------------------------------------------------
POSE_LANDMARK_VIS_THRESH = 0.50
BODY_DENSITY_G_CM3 = 1.04
REF_ARUCO_MM = 200
WHO_CSV = "who_len_height_sd.csv"

# ---------- DATA CLASSES ----------------------------------------------------------
@dataclass
class LandmarkSet:
    head: Tuple[int,int]; ankle: Tuple[int,int]
    shoulders: Tuple[int,int]; hips: Tuple[int,int]

@dataclass
class AnthropoResult:
    height_cm: float
    weight_kg: float
    haz: float
    stunted: bool
    debug: Dict

# ---------- UTILS -----------------------------------------------------------------
def load_who_table() -> pd.DataFrame:
    df = pd.read_csv(WHO_CSV)
    return df.set_index(['sex','age_mo'])

def aruco_scale(image) -> Optional[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, det)
    if ids is None: return None
    px_size = np.mean([cv2.norm(c[0][0]-c[0][2]) for c in corners])
    return (REF_ARUCO_MM/10) / px_size

def green_mat_scale(image) -> Optional[float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35,50,50), (85,255,255))
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    y0,y1 = c[:,:,1].min(), c[:,:,1].max()
    px = abs(y1-y0)
    return 100/px if px>0 else None

# ---------- POSE-BASED HEIGHT -----------------------------------------------------
def pose_landmarks(image) -> Optional[LandmarkSet]:
    try:
        import mediapipe as mp
    except ImportError:
        return None
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: return None
        lm = res.pose_landmarks.landmark
        if lm[mp.solutions.pose.PoseLandmark.NOSE].visibility < POSE_LANDMARK_VIS_THRESH:
            return None
        h,w,_ = image.shape
        def to_xy(idx): L=lm[idx]; return int(L.x*w), int(L.y*h)
        return LandmarkSet(
            head=to_xy(mp.solutions.pose.PoseLandmark.NOSE),
            ankle=min([to_xy(mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
                       to_xy(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE)], key=lambda p:p[1]),
            shoulders=(to_xy(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
                       to_xy(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)),
            hips=(to_xy(mp.solutions.pose.PoseLandmark.LEFT_HIP),
                  to_xy(mp.solutions.pose.PoseLandmark.RIGHT_HIP))
        )

def height_from_pose(lms:LandmarkSet, cm_per_px:float)->float:
    return cm_per_px * math.hypot(lms.head[0]-lms.ankle[0], lms.head[1]-lms.ankle[1])

# ---------- SILHOUETTE FALLBACK ---------------------------------------------------
def silhouette_height(image, cm_per_px)->Tuple[float, Dict]:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thr,_=cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    edge=cv2.Canny(thr,50,150)
    cnts,_=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt=max(cnts,key=cv2.contourArea)
    ys=cnt[:,:,1]; top,bottom=int(ys.min()), int(ys.max())
    height_cm=cm_per_px*(bottom-top)
    return height_cm, {"silhouette_px_top":int(top),"pix_bottom":int(bottom)}

# ---------- WEIGHT ESTIMATION -----------------------------------------------------
def weight_depth(depth_npz, cm_per_px, density=BODY_DENSITY_G_CM3)->float:
    depth = np.load(depth_npz)['arr_0']
    body_mask = depth>0
    pix_area_cm2 = (1/cm_per_px)**2
    vol_cm3 = depth[body_mask].sum()*pix_area_cm2
    return vol_cm3*density/1000

def weight_rgb_rule(height_cm, waist_px, cm_per_px)->float:
    waist_cm = waist_px*cm_per_px
    bmi_median = 16
    return bmi_median * (height_cm/100)**2 * (waist_cm/(0.5*height_cm))**0.3

# ---------- HAZ CALC --------------------------------------------------------------
def compute_haz(height_cm, age_mo, sex, who_df)->Tuple[float,bool]:
    med = who_df.loc[(sex,age_mo),'median']
    sd  = who_df.loc[(sex,age_mo),'sd']
    haz = (height_cm - med)/sd
    return haz, haz < -2.0

# ---------- MAIN ------------------------------------------------------------------
def process(image, depth_file, age_mo, sex, manual_cm_per_px: Optional[float]=None)->AnthropoResult:
    t0 = time.time()
    who_df = load_who_table()
    cm_per_px = aruco_scale(image) or green_mat_scale(image) or manual_cm_per_px
    if cm_per_px is None:
        raise ValueError("Could not determine scale (cm per px).")
    debug={"cm_per_px":cm_per_px}
    lms = pose_landmarks(image)
    if lms:
        height_cm = height_from_pose(lms, cm_per_px)
        waist_px  = abs(lms.shoulders[0][0]-lms.shoulders[1][0])
        method="pose"
    else:
        height_cm,sil_dbg = silhouette_height(image, cm_per_px)
        waist_px  = None
        method="silhouette"
        debug.update(sil_dbg)
    debug["method"]=method
    if depth_file:
        weight_kg = weight_depth(depth_file, cm_per_px)
    elif waist_px:
        weight_kg = weight_rgb_rule(height_cm, waist_px, cm_per_px)
    else:
        weight_kg = np.nan
    haz, stunted = compute_haz(height_cm, age_mo, sex, who_df)
    debug["timing_s"]=round(time.time()-t0,3)
    return AnthropoResult(height_cm, weight_kg, haz, stunted, debug)
