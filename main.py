from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
from calibration import aruco, green_mat
from model.comvistunt import draw_landmarks, get_landmarks

app = FastAPI()
UPLOAD_FOLDER = 'uploads'
landmarks = f"{UPLOAD_FOLDER}/landmark"
calibration = f"{UPLOAD_FOLDER}/calibration"
landmarks_result = f"{UPLOAD_FOLDER}/landmark/landmark_result"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(landmarks, exist_ok=True)
os.makedirs(calibration, exist_ok=True)
os.makedirs(landmarks_result, exist_ok=True)

@app.get("/")
async def root():
    return {
            "message": "Hello World",
            "status": 200

            }

@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)

@app.get("/landmark")
async def landmark():
    return JSONResponse(content={"message": "pong"}, status_code=200)

@app.post("/calibrate/aruco")
async def calibrate(image: UploadFile = File(...)):
    try:
        image.filename = "aruco.png"
        file_path = os.path.join(calibration, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        result, file_path = aruco(file_path)

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

@app.post("/get_landmark")
async def obtain_landmark(image: UploadFile = File(...)):
    try:
        image.filename = "landmark.jpg"
        file_path = os.path.join(landmarks, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
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