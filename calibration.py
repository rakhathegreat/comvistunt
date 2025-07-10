import cv2
import numpy as np
from typing import Optional 
from config_manager import get_config, set_config

def aruco(image) -> Optional[float]:
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(det, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        return None
    
    px_size = np.mean([cv2.norm(c[0][0]-c[0][2]) for c in corners])
    return float((get_config("REF_ARUCO_MM")/10) / px_size)

def green_mat(image) -> Optional[float]:
    image = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    real_length_cm = get_config("REAL_LENGTH_CM")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            object_length_px = max(w, h)
            cm_per_px = real_length_cm / object_length_px

            return cm_per_px

