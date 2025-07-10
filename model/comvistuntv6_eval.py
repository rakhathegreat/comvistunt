"""
comvistunt_v65_eval.py – Build 7.0 (evaluation‑enhanced)
============================================================
Merged & extended version of *comvistunt_v6_final.py* (a.k.a. v6.5 in
internal notes). Adds modules to **benchmark** the automatic height,
weight and stunting prediction pipeline against previously‑published
studies‐‑notably the Risfendra *et al.* IEEE Access 2025 work‑‑or any
user‑supplied ground‑truth values.

Major additions
───────────────
1. **Evaluation dashboard**  
   • Pre‑loaded datasets extracted from Risfendra dolls (38 cm & 49 cm)
     and 12 real‑infant samples.  
   • On‑the‑fly metrics: MAE, RMSE, MAPE, R², accuracy band (≤±2 cm).
2. **Custom ground‑truth input**  
   • After measuring a new child photo, user can key in reference
     height/weight to instantly view errors.
3. **Results export** – one‑click CSV download of predictions + errors
   for external analysis.

How to use
──────────
* **Standard measurement**: upload image ➝ get height, weight, HAZ.
* **Compare with research**: tick *"Enable comparison"* in sidebar,
  choose built‑in dataset or *"Manual entry"* to supply your own ground
  truth ➝ metrics appear below.
* **Export**: press *Download CSV* under metrics table.

Dependencies: `streamlit, numpy, pandas, scikit_learn` (for R² calc),
plus optional `ultralytics`, `opencv‑python`, `mediapipe`, `Pillow` as
in the original pipeline.
"""
from __future__ import annotations

import streamlit as st, numpy as np, pandas as pd, cv2, cv2.aruco as aruco, mediapipe as mp
from typing import Tuple, Optional, List, Dict
from PIL import Image
from io import BytesIO
from sklearn.metrics import r2_score
import base64, json, os

# ────────────────────────────────────────────────────────────────────────────
# 🔄 ORIGINAL DETECTION PIPELINE (UNTOUCHED LOGIC) ─────────────────────────
# ────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ModuleNotFoundError:
    _YOLO_AVAILABLE = False

# (The entire detection pipeline from v6.5 is kept verbatim but wrapped
# inside a function so we can call it programmatically during batch eval)

from math import isfinite, sqrt

# === WHO tables, HAZ, model caches – identical to v6.5 =====================
_BOYS_MED = [49.9,54.7,58.4,61.4,63.9,65.9,67.6,69.2,70.6,72.0,73.3,74.5,75.7,
             76.9,78.0,79.1,80.2,81.2,82.3,83.2,84.2,85.1,86.0,87.0,87.9,88.8,
             89.6,90.5,91.3,92.1,92.9,93.7,94.4,95.2,95.9,96.6,97.3,98.0,98.7,
             99.3,99.9,100.6,101.2,101.8,102.4,103.0,103.6,104.2,104.8,105.3,
             105.9,106.4,107.0,107.5,108.0,108.5,109.0,109.5,110.0,110.5,111.0]
_GIRL_MED = [49.1,53.7,57.1,59.8,62.1,64.0,65.7,67.3,68.7,70.1,71.5,72.8,74.0,
             75.2,76.4,77.5,78.6,79.7,80.7,81.7,82.7,83.7,84.6,85.5,86.4,87.3,
             88.2,89.1,89.9,90.7,91.4,92.2,92.9,93.6,94.3,95.0,95.7,96.3,97.0,
             97.6,98.2,98.8,99.4,100.0,100.6,101.2,101.7,102.3,102.8,103.3,
             103.9,104.4,104.9,105.4,105.9,106.4,106.9,107.4,107.9,108.4,108.9]
_WHO_TABLE = {"male":{"med":_BOYS_MED}, "female":{"med":_GIRL_MED}}
_STUNT_LABELS = [(-2.0,"Severely stunted","#e02401"),(-1.0,"Stunted","#ff9e00"),(1.0,"Normal","#26a269"),(99,"Tall","#3182ce")]

def haz(h: float, age_mo: int, sex: str):
    key = "male" if sex.lower().startswith('m') else "female"
    med = _WHO_TABLE[key]["med"][int(np.clip(age_mo,0,59))]
    sd = max(med*0.05,1)
    z = (h-med)/sd
    for thr,lab,col in _STUNT_LABELS:
        if z < thr:
            return z,lab,col
    return z,_STUNT_LABELS[-1][1],_STUNT_LABELS[-1][2]

_pose_model,_seg_model=None,None

def get_pose(model_name:str):
    global _pose_model
    if not _YOLO_AVAILABLE: return None
    if _pose_model is None or _pose_model.model.model_name!=model_name:
        _pose_model = YOLO(model_name)
    return _pose_model

def get_seg():
    global _seg_model
    if not _YOLO_AVAILABLE: return None
    if _seg_model is None:
        _seg_model = YOLO("yolov8s-seg.pt")
    return _seg_model

# Scale helpers -------------------------------------------------------------
mp_pose = mp.solutions.pose

def scale_from_color_mat(bgr, cm_len, ori, low, high):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,low,high)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    px=h if ori=="Vertical" else w
    return None if px<50 else cm_len/px

# Height detectors ----------------------------------------------------------

def height_yolo(bgr,scale,model_name,conf):
    m=get_pose(model_name)
    if m is None: return None
    res=m(bgr,conf=conf,verbose=False)[0]
    if res.keypoints is None or not len(res.keypoints): return None
    kps=res.keypoints[0].cpu().numpy()
    head_y=kps[0,1]
    foot_y=max(kps[15,1],kps[16,1])
    px=foot_y-head_y
    return None if px<=0 else px*scale

def height_mediapipe(bgr,scale):
    with mp_pose.Pose(static_image_mode=True) as pose:
        res=pose.process(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: return None
        h=bgr.shape[0]
        l=res.pose_landmarks.landmark
        head_y=l[0].y*h
        foot_y=max(l[29].y,l[30].y)*h
        px=foot_y-head_y
        return None if px<=0 else px*scale

def height_from_mask(mask,scale):
    ys=np.where(mask.any(axis=1))[0]
    if ys.size==0: return None
    px=ys[-1]-ys[0]
    return px*scale

# Weight via mask area ------------------------------------------------------

def weight_area(mask,scale,coef):
    px_area=mask.sum()
    cm2=px_area*(scale**2)
    return coef*cm2,cm2

# Core predictor callable ----------------------------------------------------

def predict_anthro(np_rgb: np.ndarray,
                   pose_model_name:str="yolov8n-pose.pt",
                   conf_pose:float=0.10,
                   scale_src:str="Manual",
                   manual_cm_per_px:float|None=0.1,
                   hsv_low:Tuple[int,int,int]=(40,40,40),
                   hsv_high:Tuple[int,int,int]=(85,255,255),
                   mat_cm_len:float=100.0,
                   mat_ori:str="Vertical",
                   weight_method:str="BMI",
                   bmi:float=15.0,
                   kcoef:float=0.0012,
                   age_mo:int=12,
                   sex:str="Male") -> Dict:
    """Headless API used by both UI and batch evaluation."""
    bgr=cv2.cvtColor(np_rgb,cv2.COLOR_RGB2BGR)
    # 1️⃣ SCALE ---------------------------------------------------------
    if scale_src=="Colored mat (HSV)":
        scale=scale_from_color_mat(bgr,mat_cm_len,mat_ori,hsv_low,hsv_high)
    else:
        scale=manual_cm_per_px
    if not scale:
        raise RuntimeError("Scale not found – mat not detected & no manual value.")
    # 2️⃣ MASK ----------------------------------------------------------
    mask=None
    seg=get_seg()
    if seg:
        res=seg(bgr,conf=0.1,verbose=False)[0]
        if res.masks is not None and len(res.masks):
            idx=int(np.argmax(res.masks.data.sum(dim=(1,2)).cpu().numpy()))
            mask=res.masks.data[idx].cpu().numpy().astype(np.uint8)
    # 3️⃣ HEIGHT --------------------------------------------------------
    h_cm=(height_yolo(bgr,scale,pose_model_name,conf_pose) or
           height_mediapipe(bgr,scale) or
           (height_from_mask(mask,scale) if mask is not None else None))
    if h_cm is None or not (20<=h_cm<=130):
        raise RuntimeError("Height detection failed or implausible result.")
    # 4️⃣ WEIGHT --------------------------------------------------------
    if weight_method=="BMI":
        w_kg=bmi*(h_cm/100)**2
        dbg={"method":"BMI","BMI":bmi}
    else:
        if mask is None:
            w_kg=15*(h_cm/100)**2
            dbg={"method":"BMI‑fallback"}
        else:
            w_kg,area_cm2=weight_area(mask,scale,kcoef)
            dbg={"method":"Area","κ":kcoef,"mask_area_cm2":area_cm2}
    # 5️⃣ HAZ -----------------------------------------------------------
    z,cat,col=haz(h_cm,age_mo,sex)
    return {"height_cm":h_cm,
            "weight_kg":w_kg,
            "haz":z,
            "haz_cat":cat,
            "haz_color":col,
            "scale_cm_per_px":scale,
            **dbg}

# ────────────────────────────────────────────────────────────────────────────
# 📊 BENCHMARK DATA FROM RISFENDRA 2025 IEEE ACCESS (extracted manually)
# ────────────────────────────────────────────────────────────────────────────

RISFENDRA_DATA: Dict[str, Dict[str, List[float]]] = {
    "Doll 38 cm": {
        "gt": [38,38,38,38,38,38],
        "pred": [37.71,38.24,37.61,37.71,37.78,37.97]
    },
    "Doll 49 cm": {
        "gt": [49,49,49,49,49,49],
        "pred": [48.36,48.83,49.27,49.16,49.37,49.38]
    },
    "Real infants (n=12)": {
        "gt": [78,71,61,60,87,63,92.4,93,65,54,85.1,68],
        "pred": [79.30,69.70,59.70,58.46,88.18,60.05,95.95,91.19,66.12,53.53,86.84,67.98]
    }
}

# Evaluation helpers --------------------------------------------------------

def compute_metrics(gt: List[float], pred: List[float]) -> Dict[str, float]:
    a=np.array(gt); p=np.array(pred)
    diff=p-a
    mae=float(np.mean(np.abs(diff)))
    rmse=float(np.sqrt(np.mean(diff**2)))
    mape=float(np.mean(np.abs(diff)/a)*100)
    r2=float(r2_score(a,p)) if len(a)>1 else np.nan
    acc_band=float(np.mean(np.abs(diff)<=2)*100)  # ≤2 cm accuracy
    return {"MAE (cm)":mae,
            "RMSE (cm)":rmse,
            "MAPE (%)":mape,
            "R²":r2,
            "Acc ≤2 cm (%)":acc_band}

# Convert metrics dict to styled dataframe ----------------------------------

def metrics_df(metrics: Dict[str,float]):
    df=pd.DataFrame(metrics,index=[0]).T.rename(columns={0:"Value"})
    return df

# Utility: CSV download -----------------------------------------------------

def get_table_download_link(df: pd.DataFrame, filename: str="results.csv"):
    csv=df.to_csv(index=False).encode()
    b64=base64.b64encode(csv).decode()
    href=f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# ────────────────────────────────────────────────────────────────────────────
# 🌐 STREAMLIT UI – MAIN APP ───────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Anthro‑Vision 7.0 – with Benchmark",layout="wide")

st.title("Anthro‑Vision 7.0 🍼📏 – Auto Height/Weight • Stunting • Evaluation")

# Sidebar ────────────────────────────────────────────────────────────────
sid=st.sidebar
sid.header("🏗️ Input")

upl=sid.file_uploader("Upload child image",type=['jpg','jpeg','png'])

pose_model_name=sid.selectbox("Pose model",["yolov8n-pose.pt","yolov8s-pose.pt"],index=0)
conf_pose=sid.slider("Pose confidence",0.05,0.5,0.10,0.01)

scale_src=sid.selectbox("Scale source",["Colored mat (HSV)","Manual"])
if scale_src=="Colored mat (HSV)":
    hmin=sid.slider("Hue min",0,179,40)
    hmax=sid.slider("Hue max",0,179,85)
    smin=sid.slider("Sat min",0,255,40)
    vmin=sid.slider("Val min",0,255,40)
    cm_len=sid.number_input("Mat length (cm)",30.0,200.0,100.0)
    ori=sid.selectbox("Mat orientation",["Vertical","Horizontal"],0)
else:
    hmin=hmax=smin=vmin=0; cm_len=100; ori="Vertical"

if scale_src=="Manual":
    cm_per_px_input=sid.text_input("cm per pixel","0.10")
    try:
        manual_scale=float(cm_per_px_input.replace(',','.'))
    except ValueError:
        manual_scale=None
else:
    manual_scale=None

sid.subheader("Weight method")
wm=sid.radio("",["BMI","Area"],index=0)
if wm=="BMI":
    bmi=sid.slider("Assumed BMI",10.0,25.0,15.0,0.1)
    kcoef=None
else:
    kcoef=sid.number_input("κ (kg/cm²)",0.0003,0.0030,0.0012,0.00005,format="%.4f")
    bmi=None

age=sid.number_input("Age (months)",0,60,12)
sex=sid.radio("Sex",["Male","Female"],0)

debug=sid.checkbox("Show debug")

# Evaluation toggle --------------------------------------------------------

sid.header("📏 Evaluation")
compare_toggle=sid.checkbox("Enable comparison with ground truth / studies")

eval_mode="None"
selected_ds=""
manual_gt_height=None
if compare_toggle:
    eval_mode=sid.radio("Mode",["Use Risfendra dataset","Manual entry"],index=0)
    if eval_mode=="Use Risfendra dataset":
        selected_ds=sid.selectbox("Choose dataset",list(RISFENDRA_DATA.keys()))
    else:
        manual_gt_height=sid.number_input("Ground‑truth height (cm)",20.0,130.0,70.0)

# Main logic ──────────────────────────────────────────────────────────────

if upl is None:
    st.info("Upload an image to start.")
    st.stop()

img=Image.open(upl).convert("RGB")
np_img=np.array(img)

try:
    res=predict_anthro(np_img,
                       pose_model_name=pose_model_name,
                       conf_pose=conf_pose,
                       scale_src=scale_src,
                       manual_cm_per_px=manual_scale,
                       hsv_low=(hmin,smin,vmin),
                       hsv_high=(hmax,255,255),
                       mat_cm_len=cm_len,
                       mat_ori=ori,
                       weight_method=wm,
                       bmi=bmi if bmi is not None else 15.0,
                       kcoef=kcoef if kcoef is not None else 0.0012,
                       age_mo=age,
                       sex=sex)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# Display primary results --------------------------------------------------

c1,c2,c3=st.columns(3)
with c1: st.metric("Height (cm)",f"{res['height_cm']:.1f}")
with c2: st.metric("Weight (kg)",f"{res['weight_kg']:.1f}")
with c3:
    st.markdown(f"<span style='padding:4px 8px;border-radius:6px;background:{res['haz_color']};color:#fff'>HAZ {res['haz']:.2f} – {res['haz_cat']}</span>",unsafe_allow_html=True)

st.image(img,caption="Input",use_column_width=True)

if debug:
    st.expander("Debug JSON").json(res)

# ─ Evaluation panel ─────────────────────────────────────────────────────

if compare_toggle:
    st.subheader("📈 Evaluation vs Ground Truth")
    if eval_mode=="Use Risfendra dataset":
        ds=RISFENDRA_DATA[selected_ds]
        gt=ds["gt"]
        pred=ds["pred"]
        # Append current sample to list for broader comparison
        gt_all=gt+[gt[0]]  # dummy – we don't know current GT
        pred_all=pred+[res['height_cm']]
        m=compute_metrics(gt_all,pred_all)
        st.markdown(f"**Dataset:** {selected_ds} + *current sample (no GT)*")
    else:
        gt=[manual_gt_height]
        pred=[res['height_cm']]
        m=compute_metrics(gt,pred)
    st.table(metrics_df(m))
    if eval_mode=="Use Risfendra dataset":
        # Show side‑by‑side dataframe
        df=pd.DataFrame({"GT (cm)":gt_all,"Pred (cm)":pred_all})
    else:
        df=pd.DataFrame({"GT (cm)":gt,"Pred (cm)":pred})
    st.dataframe(df,use_container_width=True)
    st.markdown(get_table_download_link(df,filename="pred_vs_gt.csv"),unsafe_allow_html=True)
