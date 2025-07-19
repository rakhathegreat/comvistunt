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

    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    file_path = "uploads/calibration/aruco.png"
    cv2.imwrite(file_path, image)

    return [float((get_config("REF_ARUCO_MM")/10) / px_size), file_path]

def green_mat(image) -> Optional[float]:
    image = cv2.imread(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([35, 40, 40])
    # upper_green = np.array([85, 255, 255])

    lower_green = np.array([75, 100, 100])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    real_length_cm = get_config("REAL_LENGTH_CM")

    # Temukan kontur dengan area terbesar
    max_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500 and area > max_area:
            max_area = area
            max_contour = contour

    # Jika ada kontur terbesar, proses dia saja
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        object_length_px = max(w, h)

        if object_length_px > 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            file_path = "uploads/calibration/green_mat.png"
            cv2.imwrite(file_path, image)
            cm_per_px = real_length_cm / object_length_px

            return [cm_per_px, file_path]
