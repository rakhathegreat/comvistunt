import cv2
import numpy as np
from calibration import aruco

# Baca gambar
image = cv2.imread('image/test/test2.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Rentang warna hijau dalam HSV
lower_green = np.array([75, 100, 100])
upper_green = np.array([90, 255, 255])

# Buat mask untuk warna hijau
mask = cv2.inRange(hsv, lower_green, upper_green)

# Temukan kontur
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Panjang sebenarnya objek hijau dalam cm
real_length_cm = 100

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
        cm_per_px = real_length_cm / object_length_px
        estimated_length_cm = object_length_px * cm_per_px

        # Gambar kotak dan label
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{estimated_length_cm:.2f} cm"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print(f"Object length: {object_length_px} px, approx: {cm_per_px}")
else:
    print("Tidak ditemukan objek hijau yang cukup besar.")

# Tampilkan gambar hasil
cv2.imshow('Detected Green Object', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image = cv2.imread('image/test.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# det = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(det, parameters)
# corners, ids, rejected = detector.detectMarkers(gray)
# px_size = np.mean([cv2.norm(c[0][0]-c[0][2]) for c in corners])
# print(f"px_size: {px_size}")

# cv2.aruco.drawDetectedMarkers(image, corners, ids)
# cv2.putText(image, f"cm per pixel: {(108/10)/px_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# cv2.imshow('test',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()