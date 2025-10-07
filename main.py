from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from calibration import aruco, green_mat
from model.comvistunt import draw_landmarks, get_landmarks, get_height, get_haz, get_weight
from config_manager import get_config, set_config
from fastapi.middleware.cors import CORSMiddleware
import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from camera import generate_frames



class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob: https://fastapi.tiangolo.com; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.jsdelivr.net/npm/; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.jsdelivr.net/npm/; "
            "connect-src *;"
        )
        return response


app = FastAPI()
app.add_middleware(CSPMiddleware)
UPLOAD_FOLDER = 'uploads'
landmarks = f"{UPLOAD_FOLDER}/landmark"
calibration = f"{UPLOAD_FOLDER}/calibration"
landmarks_result = f"{UPLOAD_FOLDER}/landmark/landmark_result"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(landmarks, exist_ok=True)
os.makedirs(calibration, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="uploads/landmark"), name="static")

@app.get("/")
async def root():
    return {
            "message": "Hello World",
            "status": 200

            }

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)

@app.post("/calibrate/aruco")
async def calibrate(image: UploadFile = File(...)):
    try:
        image.filename = "aruco.png"
        file_path = os.path.join(calibration, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        output = aruco(file_path)

        if not output or output[0] is None:
            return JSONResponse(
                status_code=404,
                content={"status": "failed", "message": "Calibration Failed. Marker not detected."}
            )
        
        result, file_path = output

        set_config("CM_PER_PX", result)
        
        return {
            "status": "success",
            "message": "Calibration Success.",
            "result": result,
            "file_path": file_path
        }
        
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
    
@app.post("/calibrate/green_mat")
async def calibrate(image: UploadFile = File(...)):
    try:
        image.filename = "green_mat.png"
        file_path = os.path.join(calibration, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        result, file_path = green_mat(file_path)

        if result is None:
            return JSONResponse(status_code=404, content={"status": "failed", "message": "Calibration Failed."})
        
        return {
            "status": "success",
            "message": "Calibration Success.",
            "result": result,
            "file_path": file_path
        }
        
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)



@app.post("/capture")
async def upload(image: UploadFile = File(...)):
    try:
        image.filename = "image.png"
        file_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(file_path)
        lms = get_landmarks(file_path)
        result = draw_landmarks(file_path, lms)


        return {
            "status": "success",
            "message": "Image Uploaded.",
            "image": result
        }
        
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

@app.post("/get_landmark")
async def obtain_landmark():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "image.jpg")
        lms = get_landmarks(file_path)
        result = draw_landmarks(file_path, lms)


        if result is None:
            return JSONResponse(status_code=404, content={"status": "failed", "message": "Can't get landmark."})
        
        return {
            "status": "success",
            "message": "Landmark Obtained.",
            "image": result
        }
        
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
    
@app.post("/analyze")
async def analyze( gender: str = Form(...), age: int = Form(...), ):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "image.png")
        lms = get_landmarks(file_path)
        height = get_height(lms ,get_config("CM_PER_PX"))
        z, label = get_haz(height, gender, age)
        weight = get_weight(height)

        if (height is None) or (z is None) or (weight is None):
            return JSONResponse(status_code=404, content={"status": "failed", "message": "Can,t analyze."})
        
        return {
            "status": "success",
            "message": "Landmark Obtained.",
            "height": height,
            "haz": z,
            "weight": weight
        }
        
    except Exception as e:
        return JSONResponse(content={"status": "failed","message": str(e)}, status_code=500)

@app.get("/video")
async def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )