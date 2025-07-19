import cv2
import numpy as np
from config_manager import get_config

image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

det = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(det, parameters)

corners, ids, rejected = detector.detectMarkers(gray)

if corners:
    px_sizes = []
    for corner in corners:
        pts = corner[0]
        side_lengths = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        px_sizes.append(np.mean(side_lengths))
    px_size = np.mean(px_sizes)

    result = (get_config("REF_ARUCO_MM") / 10) / px_size 

    print("cm per pixel:", result, px_size, result * px_size)

    # Gambar bounding box manual
    for i, corner in enumerate(corners):
        pts = corner[0].astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        if ids is not None:
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(image, f"ID: {ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
else:
    print("Marker tidak terdeteksi.")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
